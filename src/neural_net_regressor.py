import time
import sys
import atheris

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from sklearn.datasets import load_iris


from concrete.ml.sklearn import NeuralNetRegressor as c_nnr
from sklearn.neural_network import MLPRegressor as nnr

# # Get dataset
# X, y = make_regression(n_samples=100, n_features=5, noise=1, random_state=42)

X, y = load_iris(return_X_y=True)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = y_train.reshape(-1, 1)


params = {
    "module__n_layers": 3,
    "module__n_a_bits": 3,
    "module__n_accum_bits": 8,
    "module__n_outputs": 3,
    "module__input_dim": X.shape[1],
    "module__activation_function": nn.Sigmoid,
    "max_epochs": 1000,
    "verbose": 0,
}

# Model training
model = c_nnr(n_bits=11, **params)
model, sklearn_model = model.fit_benchmark(X=X_train.astype(np.float32), y=y_train.astype(np.float32))

def compare_models(input_bytes):
  
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(X.shape[1], -2.0, 1.0) for _ in range(48)]
    data = np.array(data)

    # Predict with the models
    y_pred_sklearn = sklearn_model.predict(data.astype(np.float32))
    y_pred_concrete = model.predict(data.astype(np.float32))
    
    # Compare both outputs
    assert np.allclose(y_pred_sklearn, y_pred_concrete, atol=1), f"Error: The predictions are different, scikit prediction {y_pred_sklearn}; concrete prediction {y_pred_concrete}"
   
atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
