import sys
import atheris
from concrete.ml.sklearn import ElasticNet
from utils import initialize_models, mean_absolute_percentage_error, consume_bytes

concrete_model, scikit_model, data_info = initialize_models(ElasticNet)

def compare_models(input_bytes):
    data = consume_bytes(input_bytes, data_info, margin=0.1)
    FHE_pred = concrete_model.predict(data, execute_in_fhe=True)
    prediction = scikit_model.predict(data)

    assert(mean_absolute_percentage_error(prediction, FHE_pred) < 2)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
