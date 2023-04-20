import time
import sys
import atheris
import tenseal as ts

import numpy as np
# from IPython.display import clear_output
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target


from concrete.ml.sklearn import NeuralNetRegressor
from utils import initialize_neural_models, consume_bytes, mean_absolute_percentage_error

concrete_model, scikit_model, data_info = initialize_neural_models(NeuralNetRegressor)

def compare_models(input_bytes):
  
    error_allowed = 1
    
    # Get random data to test
    data = consume_bytes(input_bytes, data_info, n_samples=66, margin=0)
    
    data = np.array(data).astype(np.float32)
  
    # Get Poisson predictions for scikit and FHE
    scikit_pred = scikit_model.predict(data)
    concre_pred = concrete_model.predict(data, execute_in_fhe=True)
    
    error = mean_absolute_percentage_error(scikit_pred, concre_pred)
    assert (error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {error}%"
    
atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
