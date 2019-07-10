from gpytorch.settings import _feature_flag


class bo_normalize_targets(_feature_flag):
    _state = False


class cylindrical_original_training_strategy(_feature_flag):
    _state = True


class cylindrical_original_prediction_strategy(_feature_flag):
    _state = True
