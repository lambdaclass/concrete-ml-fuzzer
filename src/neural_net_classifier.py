import sys
import atheris
from concrete.ml.sklearn import NeuralNetClassifier
from utils import consume_bytes, initialize_models, mean_absolute_percentage_error

concrete_model, sklearn_model, data_info = initialize_models(NeuralNetClassifier, {})

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info, n_samples=10)

    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    prediction = sklearn_model.predict(data)

    assert(mean_absolute_percentage_error(prediction, FHE_pred) < 1)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
