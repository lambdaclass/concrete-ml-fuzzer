import sys
import atheris
from concrete.ml.sklearn import LogisticRegression 
from utils import initialize_models, mean_absolute_percentage_error, consume_bytes

concrete_model, scikit_model, data_info = initialize_models(LogisticRegression)

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info, margin=0.1)
    # Run the inference, encryption and decryption is done in the background
    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Get the mean percentage error to make sure the accuaracy is around 99%
    error = mean_absolute_percentage_error(prediction, FHE_pred)
    assert(error < 1) , f"Error: The prediction accuracy compared to scikit is less than 99%: the error percentage is {error}%"

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
