import numpy as np
import sys
import atheris
from sklearn.svm import LinearSVR as lsvr
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn.svm import LinearSVR as c_lsvr

# Dataset to train
X, y = make_classification(n_samples=5, n_features=5, random_state=0)

# Split the data-set into a train and test set,
# each set is split into input and result.
input_train, _, result_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Start the scikit linearsvr model
scikit_model = lsvr()

# Start the concrete-ml linearsvr model
concrete_model = c_lsvr()

# Train the models
concrete_model.fit(input_train, result_train)
scikit_model.fit(input_train, result_train)

# Compile FHE
concrete_model.compile(input_train)

def compare_models(input_bytes):
  
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(5, 0.0, 1.0) for _ in range(15)]
    # Run the inference on encrypted inputs
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)
    
    # define the tolerance accepted in the comparison 
    tolerance = 1e-5

    # check if the two predictions are equal within the tolerance
    if np.allclose(fhe_pred, prediction, rtol=tolerance, atol=tolerance):
        assert(True)
    else:
        assert True 
        assert False, f"Error: The predictions are different within the tolerance of {tolerance}, scikit prediction {prediction}; concrete prediction {fhe_pred}"
    
    
atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()


