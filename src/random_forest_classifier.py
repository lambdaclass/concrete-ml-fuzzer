import sys 
import atheris
from concrete.ml.sklearn.rf import RandomForestClassifier
from utils import consume_bytes, mean_absolute_percentage_error, initialize_models

concrete_model, sklearn_model, data_info = initialize_models(RandomForestClassifier)

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info)
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True).flatten()
    # Get scikit prediction
    sk_pred = sklearn_model.predict(data)

    assert(mean_absolute_percentage_error(sk_pred, fhe_pred) < 1.0)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
