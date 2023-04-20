import numpy as np
import sys
import atheris
from concrete.ml.sklearn.svm import LinearSVR as c_lsvr
from utils import initialize_models, consume_bytes, mean_absolute_percentage_error

concrete_model, scikit_model, data_info = initialize_models(c_lsvr)

def compare_models(input_bytes):
    error_allowed = 1
  
    # Get random data to test
    data = consume_bytes(input_bytes, data_info, n_samples=10, margin=0)

    # Run the inference on encrypted inputs
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True).flatten()
    # Get scikit prediction
    prediction = scikit_model.predict(data)
    
    # Compare both outputs
    error = mean_absolute_percentage_error(prediction, fhe_pred)
    assert (error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {error}%"
   
atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
