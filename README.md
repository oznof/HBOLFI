# HBOLFI: High-Dimensional Bayesian Optimization for Likelihood-Free Inference

**A 2019 reimplementation of the methodology of my MSc thesis, written with PyTorch, GPyTorch, and Pyro at the start of my doctoral research.** It learns the additive structure of the discrepancy with Markov chain Monte Carlo, to make Bayesian optimization for likelihood-free inference (BOLFI; [Gutmann and Corander, 2016](https://jmlr.org/papers/v17/15-017.html)) efficient in higher dimensions.

- **Author:** Afonso Eduardo, [ORCID 0009-0003-0127-8223](https://orcid.org/0009-0003-0127-8223).
- **Thesis:** *High-dimensional Bayesian Optimization for Learning in Generative Models* (MSc in Artificial Intelligence, University of Edinburgh, 2018), [doi:10.5281/zenodo.24226042](https://doi.org/10.5281/zenodo.24226042).
- **Original thesis code (2018):** [hbolfi-msc-2018](https://github.com/oznof/hbolfi-msc-2018), kept as submitted.
- **Follow-up:** [Bayesian Optimization with Informative Covariance](https://arxiv.org/abs/2208.02704) (Eduardo and Gutmann, *Transactions on Machine Learning Research*, 2023; code: [BOIC](https://github.com/oznof/BOIC)).
- **Status:** research code from 2019, not maintained. Dependencies are not pinned, and the experiments were not rerun when the code was published here.

## Changes From the 2018 Thesis Code

| Aspect | 2018 thesis code | This reimplementation (2019) |
|---|---|---|
| Framework | GPy and NumPy | PyTorch, GPyTorch, and Pyro |
| Structure | Flat scripts | An installable package (`setup.py`), with one module per concern |
| Experiments | One runner script per problem, each repeating the same setup and loop | General runners (`experiments/lfi.py`, `experiments/baseline.py`) with the problem chosen by an option, and problems defined as simulator classes (`hbolfi/simulators/`) |
| Design | Functions and a few classes per script | Object-oriented: base classes for acquisition functions and surrogate models, with interchangeable kernels, means, priors, and samplers |
| Gradients | Written by hand | Automatic differentiation |
| Hardware | CPU | Tensors follow the device of their inputs, so the code is prepared for GPUs. All experiments here ran on CPU |
| Hyperpriors | Three priors written for GPy (inverse gamma, half-Student-t, exponential) | A library of informative hyperpriors, described below |
| Settings | Constants inside each script | Defaults and context managers in `hbolfi/settings/` |
| Tests | None | Unit tests in `tests/`, including a reproducibility test |

## Informative Hyperpriors

The hyperparameters of the Gaussian process (lengthscales, output scale, noise, and the coefficients of the quadratic mean) are given informative priors, which encode what is known about their plausible values before any data.

- **Any distribution as a prior:** `make_prior` turns a PyTorch or Pyro distribution into a GPyTorch prior, and derives the hyperparameter's constraint from the distribution's support, so the prior also sets the valid range and its transformation.
- **Distributions written for this purpose:** half-Student-t, log-Student-t (which gives the log-Cauchy), inverse gamma and shifted inverse gamma, truncated normal, shifted half-normal, and mixtures.
- **Spike-and-slab priors:** a narrow truncated normal (the spike) around an expected value, mixed with a uniform or wider truncated normal (the slab) over the allowed range. They state a preferred value without excluding the rest. They are the default priors of the cylindrical kernel's parameters alpha and beta, and an option for the coefficients of the quadratic mean.
- **Priors elicited from a range:** `invgamma_solver` finds the inverse-gamma parameters whose 1% and 99% quantiles match a lower and an upper bound. For the lengthscales, the default range of 0.0001 to 2 suits a search space of [-1, 1]^n, so lengthscales far beyond the size of the space are improbable.
- **One prior per hyperparameter, chosen by name:** `hbolfi/settings/priors.py` keeps a registry of priors for each hyperparameter, with a default (for example, half-Student-t for the noise and the output scale, and inverse gamma for the lengthscales). A context manager changes the prior for a block of code.

## Reproducibility

GPyTorch's fast approximations, such as stochastic trace estimation and iterative solvers, make results vary between runs even with the same seed. The experiment runners (`experiments/lfi.py`, `experiments/baseline.py`) and the test `tests/bayes_opt/test_reproducibility.py` turn them off explicitly, so the computations are exact and deterministic:

- `_fast_covar_root_decomposition`, `_fast_log_prob`, `_fast_solves`, and `fast_pred_var` set to off
- `num_trace_samples` set to 0
- `torch.float64` as the default type
- seeds set for PyTorch and NumPy at the start of each run (`torch.manual_seed`, `np.random.seed`)

The test checks that two runs with the same seed give the same model parameters. Running on CPU avoids the nondeterminism of some GPU operations. Results depend on the library versions, so other versions of PyTorch, GPyTorch, or the numerical libraries beneath them can give small numerical differences.

## Contents

| Path | Content |
|---|---|
| `hbolfi/bayes_opt/` | Bayesian optimization and BOLFI |
| `hbolfi/models/gp/` | Gaussian process surrogates: exact, additive, and Bayesian (hyperparameters sampled) |
| `hbolfi/kernel_discovery/additive/` | Structure learning of additive Gaussian processes: Metropolis-Hastings and Gibbs samplers |
| `hbolfi/acquisitions/` | Acquisition functions: lower confidence bound, expected improvement, and an additive version |
| `hbolfi/kernels/`, `hbolfi/means/` | Covariance functions, including a cylindrical kernel, and a quadratic mean function |
| `hbolfi/inference/` | Approximate inference: Laplace approximation, No-U-Turn Sampler, and Gaussian variational inference |
| `hbolfi/distributions/` | Distributions for informative hyperpriors, and the conversion of any distribution into a prior (`make_prior`) |
| `hbolfi/simulators/` | Test problems: quadratic-form discrepancy models (parametric and semiparametric) and benchmark functions |
| `hbolfi/settings/` | Default settings, and the prior of each hyperparameter as a context manager (`priors.py`) |
| `experiments/` | Runners for likelihood-free inference (`lfi.py`, `prepare_lfi.py`) and Bayesian optimization baselines (`baseline.py`, `run_baseline.sh`) |
| `notebooks/cylindrical/` | Benchmarks of the cylindrical kernel, described below |
| `tests/` | Unit tests |

## Cylindrical Kernel Benchmarks

`notebooks/cylindrical/cylindrical_benchmarks.ipynb` reproduces BOCK, Bayesian optimization with cylindrical kernels ([Oh, Gavves, and Welling, 2018](https://proceedings.mlr.press/v80/oh18a.html)), on that paper's benchmarks: Rosenbrock (20 dimensions), Levy (20), Hartmann6 (18), and Branin (20). It compares the final minima with those reported in the paper, and tests two settings: 2 or 20 initial points, and a fixed or randomized Sobol sequence for the starting points of the acquisition-function optimization.

- **Code:** the runs used the authors' own package, [HyperSphere](https://github.com/ChangYong-Oh/HyperSphere), through `experiments/hypersphere/run_bo.py`.
- **Data:** `notebooks/cylindrical/data/` holds the log of each run, one CSV file per run, with seeds 117 to 121 in the folder names. These logs are all the notebook reads.
- **Relevance:** this was early doctoral work on keeping Bayesian optimization away from the boundary of the search space. It led to the informative covariance functions of the follow-up paper.

## Requirements

Python 3, with the packages in `requirements.txt`: PyTorch (1.1 or later), GPyTorch (0.3.3 or later), Pyro (0.3.3 or later), NumPy, SciPy, SymPy, Matplotlib, dill, ghalton, and sampyl-mcmc. The benchmarks also need HyperSphere. Releases from 2019 are the most likely to work.

## License

MIT License. See [LICENSE](LICENSE).
