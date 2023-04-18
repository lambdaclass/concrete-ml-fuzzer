import sys
import atheris
from concrete.ml.sklearn import LinearSVC
from utils import initialize_models, mean_absolute_percentage_error, consume_bytes

concrete_model, scikit_model, data_info = initialize_models(LinearSVC)

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info)
    # Run the inference, encryption and decryption is done in the background
    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Get the mean percentage error to make sure the accuaracy is around 99%
    assert(mean_absolute_percentage_error(prediction, FHE_pred) < 1)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
