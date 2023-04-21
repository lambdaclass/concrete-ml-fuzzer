import sys
import atheris
from concrete.ml.sklearn import NeuralNetClassifier
from utils import initialize_neural_models, consume_bytes, mean_absolute_percentage_error

# Todo: this fails because it asks for a tensor of 0D or 1D and the dataset returns a multi-target
concrete_model, sklearn_model, data_info, _ = initialize_neural_models(NeuralNetClassifier)

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info, n_samples=5)

    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    prediction = sklearn_model.predict(data)

    assert(mean_absolute_percentage_error(prediction, FHE_pred) < 1)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
