import numpy as np
import sys
import atheris
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as dtr
from concrete.ml.sklearn import DecisionTreeRegressor 
from utils import initialize_models, consume_bytes, mean_absolute_percentage_error

from sklearn.base import is_classifier
from sklearn.datasets import load_iris, make_regression

concrete_model, scikit_model, data_info = initialize_models(DecisionTreeRegressor, {})

def compare_models(input_bytes):
    error_allowed = 1
    
    # Get random data to test
    data = consume_bytes(input_bytes, data_info, n_samples=10)

    # Get predictions for scikit and FHE
    scikit_pred = scikit_model.predict(data)
    concre_pred = concrete_model.predict(data, execute_in_fhe=True)
    
    # check accuracy
    error = mean_absolute_percentage_error(scikit_pred, concre_pred)
    assert (error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {error}%"
    
atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
