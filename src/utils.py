def _classification_training() -> data, restult, data_info:
    """
    Create a classification training set
    """
    return

def _regression_training():
    """
    Create a regression training set
    """
    return

def initialize_models(Model: ModelClass, *params) -> (model, model, data_info):
    """
    Initialize concrete and sklearn models. Automatically load iris or diabetes dataset
    """
    return

def consume_bytes(input_bytes: bytes, data_info, margin, n_samples) -> data:
    """
    Consume bytes and return a testing dataset for the models
    """
    return

def mean_absolute_percentage_error(y_fhe, y_sklearn) -> float:
    """
    Calculate the mean absolute percentage error to determine how much the fhe model deviates from the sklearn model.
    Values fall in the (0, 100) range, lower values represent less deviation.
    """
    return 
