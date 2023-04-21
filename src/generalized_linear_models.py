import numpy as np
import sys
import atheris
from concrete.ml.sklearn import GammaRegressor as ConcreteGammaRegressor
from concrete.ml.sklearn import PoissonRegressor as ConcretePoissonRegressor
from concrete.ml.sklearn import TweedieRegressor as ConcreteTweedieRegressor
from sklearn.linear_model import GammaRegressor as SklearnGammaRegressor
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.linear_model import TweedieRegressor as SklearnTweedieRegressor
from utils import initialize_models, initialize_glm_models, consume_bytes, mean_absolute_percentage_error

poisson_init_parameters = {
    "alpha": 1e-3,
    "max_iter": 400,
    "n_bits":12,
}

gamma_init_parameters = {
   "alpha": 10.0,
    "max_iter": 300,
    "n_bits":12,
}

tweedie_init_parameters = {
    "power": 1.9,
    "alpha": 0.1,
    "max_iter": 10000,
    "n_bits":12,
}

# Initialize the models
poisson_concrete_model, poisson_scikit_model, poisson_data_info = initialize_glm_models(ConcretePoissonRegressor, SklearnPoissonRegressor, "Poisson", poisson_init_parameters)
gamma_concrete_model, gamma_scikit_model, gamma_data_info = initialize_glm_models(ConcreteGammaRegressor, SklearnGammaRegressor, "Gamma", gamma_init_parameters)
tweedie_concrete_model, tweedie_scikit_model, tweedie_data_info = initialize_glm_models(ConcreteTweedieRegressor, SklearnTweedieRegressor, "Tweedie", tweedie_init_parameters)
    
def compare_models(input_bytes):
    error_allowed = 1
    
    # Get random data to test
    poisson_data = consume_bytes(input_bytes, poisson_data_info, n_samples=66, margin=0)
    gamma_data = consume_bytes(input_bytes, gamma_data_info, n_samples=10, margin=0)
    tweedie_data = consume_bytes(input_bytes, tweedie_data_info, n_samples=10, margin=0)
    
    # Get Poisson predictions for scikit and FHE
    poisson_scikit_pred = poisson_scikit_model.predict(poisson_data)
    poisson_concre_pred = poisson_concrete_model.predict(poisson_data, execute_in_fhe=True)
    
    # check accuracy
    poisson_error = mean_absolute_percentage_error(poisson_scikit_pred, poisson_concre_pred)
    assert (poisson_error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {poisson_error}%"
    
    # Get Gamma predictions for scikit and FHE
    gamma_scikit_pred = gamma_scikit_model.predict(gamma_data)
    gamma_concre_pred = gamma_concrete_model.predict(gamma_data, execute_in_fhe=True)
    
    # check accuracy
    gamma_error = mean_absolute_percentage_error(gamma_scikit_pred, gamma_concre_pred)
    assert (gamma_error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {gamma_error}%"
    
    # Get Tweedie predictions for scikit and FHE
    tweedie_scikit_pred = tweedie_scikit_model.predict(tweedie_data)
    tweedie_concre_pred = tweedie_concrete_model.predict(tweedie_data, execute_in_fhe=True)
    
    # check accuracy
    tweedie_error = mean_absolute_percentage_error(tweedie_scikit_pred, tweedie_concre_pred)
    assert (tweedie_error < error_allowed ), f"Error: The prediction accuracy compared to scikit is less than {100 - error_allowed}%: the error percentage is {tweedie_error}%"
    

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
