import atheris
import numpy as np
from sklearn.base import is_classifier
from sklearn.datasets import load_iris, make_regression

def _classification_training() -> (np.array, np.array, dict):
    """
    Create a classification training set
    """
    training_x, training_y = load_iris(return_X_y=True)
    data_info = {
        "dimensions": 4,
        "min_feature": min(min(minimum) for minimum in training_x),
        "max_feature": max(max(maximum) for maximum in training_x),
    }

    return (training_x, training_y, data_info)

def _regression_training() -> (np.array, np.array, dict):
    """
    Create a regression training set
    """
    training_x, training_y = make_regression(n_samples=100, n_features=5, random_state=42)
    data_info = {
        "dimensions": 5,
        "min_feature": min([ min(minimum) for minimum in training_x]),
        "max_feature": max([ max(maximum) for maximum in training_x]),
    }

    return (training_x, training_y, data_info)

def consume_bytes(input_bytes: bytes, data_info, n_samples=1000, margin=0.1) ->  np.array:
    """
    Consume bytes and return a testing dataset for the models
    input_bytes: bytes generated by atheris
    data_info: dictionary provided by initialize_models; contains information to build the random datasets
    n_samples: the size of the dataset
    margin: margin to expand the dataset range beyond the range of the training set. Recomended to use small values
    """
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [
            fdp.ConsumeFloatListInRange(
                data_info["dimensions"], 
                data_info["min_feature"] - margin, 
                data_info["max_feature"] + margin
            ) for _ in range(n_samples)
           ]
    return data

def mean_absolute_percentage_error(y_sklearn, y_fhe) -> float:
    """
    Calculate the mean absolute percentage error to determine how much the fhe model deviates from the sklearn model.
    Values fall in the (0, 100) range, lower values represent less deviation.
    """
    
    # Compute accuracy for each possible representation
    score = np.abs((y_sklearn - y_fhe) / y_sklearn)

    return np.mean(score)


def initialize_models(ModelClass, *params):
    """
    Initialize concrete and sklearn models. Automatically load iris or diabetes dataset
    """
    
    X, y, data_info = _classification_training() if is_classifier(ModelClass) else _regression_training()
    
    model = ModelClass(n_bits=11, *params)
    concrete_model, sklearn_model = model.fit_benchmark(X, y)
    concrete_model.compile(X)
    return concrete_model, sklearn_model, data_info
