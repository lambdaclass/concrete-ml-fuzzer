import sys
import atheris
from concrete.ml.sklearn import Ridge
from utils import initialize_models, mean_absolute_percentage_error, consume_bytes

concrete_model, scikit_model, data_info = initialize_models(Ridge)

def compare_models(input_bytes):
    # Run the inference, encryption and decryption is done in the background
    data = consume_bytes(input_bytes, data_info, margin=0.1)
    # Get scikit prediction
    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get the mean percentage error to make sure the accuaracy is around 98%
    prediction = scikit_model.predict(data)

    assert(mean_absolute_percentage_error(prediction, FHE_pred) < 2)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
