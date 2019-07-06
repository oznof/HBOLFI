import argparse
import os.path
import pickle
import time
import csv
from datetime import datetime
from pathlib import Path

from HyperSphere.BO.acquisition.acquisition_maximization import suggest, optimization_candidates
from HyperSphere.BO.acquisition.acquisition_maximization import optimization_init_points, deepcopy_inference
from HyperSphere.GP.models.gp_regression import GPRegression
from HyperSphere.GP.kernels.modules.matern52 import Matern52
from HyperSphere.GP.kernels.modules.radialization import RadializationKernel
from HyperSphere.GP.inference.inference import Inference
from HyperSphere.BO.shadow_inference.inference_sphere_origin import ShadowInference as origin_ShadowInference
from HyperSphere.feature_map.modules.kumaraswamy import Kumaraswamy
from HyperSphere.feature_map.functionals import sphere_bound
from HyperSphere.test_functions.benchmarks import *

import numpy as np


def step(self):
    """ Perform a slice sample step """
    dims = self.state.tovector().shape[0]
    if self.compwise:
        ordering = np.arange(dims)
        np.random.shuffle(ordering)
        new_x = self.state.tovector().copy()
        for d in ordering:
            direction = np.zeros((dims))
            direction[d] = 1.0
            new_x = self.direction_slice(direction, new_x)
    else:
        direction = np.random.randn(dims)
        direction = direction / np.sqrt(np.sum(direction ** 2))
        new_x = self.direction_slice(direction, self.state.tovector())

    self.state = self.state.fromvector(new_x)
    self._sampled += 1
    return self.state

import sampyl
sampyl.samplers.slice.Slice.step = step


def BO(func, ndim, geometry, n_eval=200, ard=False, warping=False, exp_dir='', n_init_eval=20, seed=117,
       fix_sobol=False):

    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_conf_str = geometry
    if geometry == 'sphere':
        assert (not ard)
        exp_conf_str += '_warping' if warping else ''
        radius_input_map = Kumaraswamy(ndim=1, max_input=ndim ** 0.5) if warping else None
        model = GPRegression(kernel=RadializationKernel(max_power=3, search_radius=ndim ** 0.5,
                                                        radius_input_map=radius_input_map))
        inference_method = origin_ShadowInference
        exp_conf_str += '_origin'
        bnd = sphere_bound(ndim ** 0.5)
    elif geometry == 'cube':
        exp_conf_str += '_ard' if ard else ''
        model = GPRegression(kernel=Matern52(ndim=ndim, ard=ard))
        inference_method = Inference
        bnd = (-1, 1)
    else:
        raise RuntimeError("Invalid geometry.")
    if fix_sobol:
        exp_conf_str += '_fix_sobol'
    dir_list = [elm for elm in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, elm))]
    folder_name = func.__name__ + '_D' + str(ndim) + '_' + exp_conf_str + '_' + datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
    os.makedirs(os.path.join(exp_dir, folder_name))
    log_filename = os.path.join(exp_dir, folder_name, 'log.csv')
    model_filename = os.path.join(exp_dir, folder_name, 'model.pt')
    data_config_filename = os.path.join(exp_dir, folder_name, 'data_config.pkl')
    # generate points in [-1, 1]^D, it will include origin as the first point in the set
    x_input = torch.quasirandom.SobolEngine(ndim).draw(n_init_eval)
    x_input = x_input * 2 - 1
    output = torch.zeros(n_init_eval, 1)
    for i in range(n_init_eval):
        output[i] = func(x_input[i])

    time_list = []
    pred_mean_list = []
    pred_std_list = []
    pred_var_list = []
    pred_stdmax_list = []
    pred_varmax_list = []

    inference = inference_method((x_input, output), model)
    inference.init_parameters()
    inference.sampling(n_sample=1, n_burnin=99, n_thin=1)

    ignored_variable_names = ['i', 'key', 'value', 'log_filename', 'data_config_file', 'dir_list', 'folder_name',
                              'model_filename', 'data_config_filename', 'kernel', 'model', 'inference']
    stored_variable_names = set(locals().keys()).difference(set(ignored_variable_names))

    torch.save(model, model_filename)
    stored_variable = dict()
    for key in stored_variable_names:
        stored_variable[key] = locals()[key]
    with open(data_config_filename, 'wb') as f:
        pickle.dump(stored_variable, f)
    print('Experiment based on data in %s' % os.path.split(model_filename)[0])

    for jj in range(n_eval):
        start_time = time.time()
        inference = inference_method((x_input, output), model)

        reference, ref_ind = torch.min(output, 0)
        reference = reference.item()
        gp_hyper_params = inference.sampling(n_sample=10, n_burnin=0, n_thin=1)
        inferences = deepcopy_inference(inference, gp_hyper_params)

        x0_cand = optimization_candidates(x_input, output, -1, 1, fix_sobol=fix_sobol)
        x0, sample_info = optimization_init_points(x0_cand, reference=reference, inferences=inferences)
        _out = suggest(x0=x0, reference=reference, inferences=inferences, bounds=bnd)
        next_x_point, pred_mean, pred_std, pred_var, pred_stdmax, pred_varmax = _out

        time_list.append(time.time() - start_time)
        pred_mean_list.append(pred_mean.item())
        pred_std_list.append(pred_std.item())
        pred_var_list.append(pred_var.item())
        pred_stdmax_list.append(pred_stdmax.item())
        pred_varmax_list.append(pred_varmax.item())

        x_input = torch.cat([x_input, next_x_point], 0)
        output = torch.cat([output, func(x_input[-1]).resize(1, 1)])

        min_ind = torch.min(output, 0)[1]
        min_loc = x_input[min_ind]
        min_val = output[min_ind].item()
        with torch.no_grad():
            dist_to_min = (x_input[-1] - min_loc).norm().item()
            out_of_box = torch.sum((torch.abs(x_input[-1]) > 1)).item()

        res_dict = dict(iter=jj+1, time=np.around(time_list[-1], 3),
                        curr_min=min_val, min_ind=min_ind.item(), min_loc_r=min_loc.norm().item(),
                        output=output[-1].item(), input_r=x_input[-1].norm().item(), out_of_box=out_of_box,
                        pred_mean=pred_mean_list[-1], pred_std=pred_std_list[-1],
                        input=x_input[-1].detach().numpy(), min_loc=min_loc.detach().numpy(),
                        dist_to_min=dist_to_min)
        write_mode = 'a' if jj > 0 else 'w'
        with open(log_filename, write_mode) as f:
            w = csv.DictWriter(f, res_dict.keys())
            if jj == 0:
                w.writeheader()
            w.writerow(res_dict)
        # avoid cluttering stdout
        res_dict.pop('input')
        res_dict.pop('min_loc')
        print('')
        print(res_dict)

        torch.save(model, model_filename)
        stored_variable = dict()
        for key in stored_variable_names:
            stored_variable[key] = locals()[key]
        with open(data_config_filename, 'wb') as f:
            pickle.dump(stored_variable, f)

    print('Experiment based on data in %s' % os.path.split(model_filename)[0])
    return os.path.split(model_filename)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Optimization runner')
    parser.add_argument('-g', '--geometry', dest='geometry', default='sphere', help='cube/sphere')
    parser.add_argument('--warping', dest='warping', action='store_true', default=True)  # always True
    parser.add_argument('--ard', dest='ard', action='store_true', default=False)

    parser.add_argument('-f', '--func', dest='func_name', default='levy')
    parser.add_argument('-d', '--dim', dest='ndim', type=int, default=20)
    parser.add_argument('-e', '--n_eval', dest='n_eval', type=int, default=200)
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=117)
    parser.add_argument('-fix', '--fix_sobol', dest='fix_sobol', action='store_true', default=False)

    parser.add_argument('--exp_dir', type=str, nargs='?', default="~/storage/Hypersphere/", help="Results dir.")

    args = parser.parse_args()

    settings = vars(args)
    settings['exp_dir'] = settings['exp_dir'].replace('~', str(Path.home()))

    func = globals()[settings.pop('func_name')]
    settings['func'] = func
    print(BO(**vars(args)))
