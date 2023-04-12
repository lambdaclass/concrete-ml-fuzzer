import numpy as np
import sys
import atheris
import time
from collections import defaultdict
from timeit import default_timer as timer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import GammaRegressor as SklearnGammaRegressor
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.linear_model import TweedieRegressor as SklearnTweedieRegressor
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from concrete.ml.sklearn import GammaRegressor as ConcreteGammaRegressor
from concrete.ml.sklearn import PoissonRegressor as ConcretePoissonRegressor
from concrete.ml.sklearn import TweedieRegressor as ConcreteTweedieRegressor

# Getting the original data set containing the risk features
# Link: https://www.openml.org/d/41214
risks_data, _ = fetch_openml(
    data_id=41214, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
)

# Getting the data set containing claims amount
# Link: https://www.openml.org/d/41215
claims_data, _ = fetch_openml(
    data_id=41215, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
)

# Set IDpol as index
risks_data["IDpol"] = risks_data["IDpol"].astype(int)
risks_data.set_index("IDpol", inplace=True)

# Grouping claims mounts together if they are associated with the same policy
claims_data = claims_data.groupby("IDpol").sum()

# Merging the two sets over policy IDs
data = risks_data.join(claims_data, how="left")

# Only keeping the first 100 000 for faster running time
data = data.head(100000)

# Filtering out unknown claim amounts
data["ClaimAmount"].fillna(0, inplace=True)

# Filtering out claims with zero amount, as the severity (gamma) model
# requires strictly positive target values
data.loc[(data["ClaimAmount"] == 0) & (data["ClaimNb"] >= 1), "ClaimNb"] = 0

# Removing unreasonable outliers
data["ClaimNb"] = data["ClaimNb"].clip(upper=4)
data["Exposure"] = data["Exposure"].clip(upper=1)
data["ClaimAmount"] = data["ClaimAmount"].clip(upper=200000)

log_scale_transformer = make_pipeline(FunctionTransformer(np.log, validate=False), StandardScaler())

linear_model_preprocessor = ColumnTransformer(
    [
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        ("binned_numeric", KBinsDiscretizer(n_bins=10), ["VehAge", "DrivAge"]),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        (
            "onehot_categorical",
            OneHotEncoder(sparse=False),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
    ],
    remainder="drop",
)

x = linear_model_preprocessor.fit_transform(data)

# Creating target values for Poisson
data["Frequency"] = data["ClaimNb"] / data["Exposure"]

# Creating target values for Gamma
data["AvgClaimAmount"] = data["ClaimAmount"] / np.fmax(data["ClaimNb"], 1)

# Creating target values for Tweedie
# Insurances companies are interested in modeling the Pure Premium, that is the expected total
# claim amount per unit of exposure for each policyholder in their portfolio
data["PurePremium"] = data["ClaimAmount"] / data["Exposure"]


# Split the data-set into a train and test set,
# each set is split into input and result.
train_data, test_data, x_train_data, x_test_data = train_test_split(data, x, test_size=0.2, random_state=42)

gamma_mask_train = train_data["ClaimAmount"] > 0
gamma_mask_test = test_data["ClaimAmount"] > 0

parameters_glms = {
    "Poisson": {
        "sklearn": SklearnPoissonRegressor,
        "concrete": ConcretePoissonRegressor,
        "init_parameters": {
            "alpha": 1e-3,
            "max_iter": 400,
        },
        "fit_parameters": {
            "X": x_train_data,
            "y": train_data["Frequency"],
            "sample_weight": train_data["Exposure"],
        },
        "x_test": x_test_data,
        "score_parameters": {
            "y_true": test_data["Frequency"],
            "sample_weight": test_data["Exposure"],
        },
        "deviance": mean_poisson_deviance,
    },
    "Gamma": {
        "sklearn": SklearnGammaRegressor,
        "concrete": ConcreteGammaRegressor,
        "init_parameters": {
            "alpha": 10.0,
            "max_iter": 300,
        },
        "fit_parameters": {
            "X": x_train_data[gamma_mask_train],
            "y": train_data[gamma_mask_train]["AvgClaimAmount"],
            "sample_weight": train_data[gamma_mask_train]["ClaimNb"],
        },
        "x_test": x_test_data[gamma_mask_test],
        "score_parameters": {
            "y_true": test_data[gamma_mask_test]["AvgClaimAmount"],
            "sample_weight": test_data[gamma_mask_test]["ClaimNb"],
        },
        "deviance": mean_gamma_deviance,
    },
    "Tweedie": {
        "sklearn": SklearnTweedieRegressor,
        "concrete": ConcreteTweedieRegressor,
        "init_parameters": {
            "power": 1.9,
            "alpha": 0.1,
            "max_iter": 10000,
        },
        "fit_parameters": {
            "X": x_train_data,
            "y": train_data["PurePremium"],
            "sample_weight": train_data["Exposure"],
        },
        "x_test": x_test_data,
        "score_parameters": {
            "y_true": test_data["PurePremium"],
            "sample_weight": test_data["Exposure"],
            "power": 1.9,
        },
        "deviance": mean_tweedie_deviance,
    },
}

def compare_models(input_bytes):
    
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(74, 0.0, 1.0) for _ in range(15)]
    
    
    for glm, parameters_glm in parameters_glms.items():
        # Retrieve the regressors
        sklearn_class = parameters_glm["sklearn"]
        concrete_class = parameters_glm["concrete"]

        # Instantiate the models
        init_parameters = parameters_glm["init_parameters"]
        sklearn_glm = sklearn_class(**init_parameters)
        concrete_glm = concrete_class(**init_parameters)

        # Fit the models
        fit_parameters = parameters_glm["fit_parameters"]
        sklearn_glm.fit(**fit_parameters)
        concrete_glm.fit(**fit_parameters)

        x_train_subset = fit_parameters["X"][:100]
        # Compile the Concrete-ML in FHE
        
        circuit = concrete_glm.compile(x_train_subset)

        # Generate the key
        sys.stdout.flush()

        time_begin = time.time()
        circuit.client.keygen(force=False)
                
        # Compute the predictions using sklearn (floating points, in the clear)
        sklearn_predictions = sklearn_glm.predict(data)

        # Compute the predictions using COncrete-ML (in FHE)
        concrete_predictions = concrete_glm.predict(
            data,
            execute_in_fhe=True,
        ).flatten()
        print(sklearn_predictions)
        print(concrete_predictions)

        # Compute the deviance scores
        mean_deviance = parameters_glm["deviance"]
        score_parameters = parameters_glm["score_parameters"]
        assert np.allclose(sklearn_predictions, concrete_predictions, atol=1), f"Error: The predictions are different, scikit prediction {sklearn_predictions}; concrete prediction {concrete_predictions}"
        # sklearn_score = mean_deviance(y_pred=sklearn_predictions, **score_parameters)
        # concrete_score = mean_deviance(concrete_predictions, **score_parameters)

        # # Measure the error of the FHE quantized model with respect to the clear scikit-learn
        # # float model
        # score_difference = abs(concrete_score - sklearn_score) * 100 / sklearn_score
        # assert score_difference < 1


atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()

def compare_models(input_bytes):
    
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(74, 0.0, 1.0) for _ in range(15)]
    
    # Instantiate the models
    sklearn_glm = SklearnGammaRegressor(**init_parameters)
    concrete_glm = ConcreteGammaRegressor(**init_parameters)
    
    # Fit the models
    sklearn_glm.fit(**fit_parameters)
    concrete_glm.fit(**fit_parameters)
    
    # Compute the predictions using sklearn (floating points, in the clear)
    sklearn_predictions = sklearn_glm.predict(data)

    # Compute the predictions using COncrete-ML (in FHE)
    concrete_predictions = concrete_glm.predict(
        data,
        execute_in_fhe=True,
    ).flatten()
    print(sklearn_predictions)
    print(concrete_predictions)
    
    assert np.allclose(sklearn_predictions, concrete_predictions, atol=1), f"Error: The predictions are different, scikit prediction {sklearn_predictions}; concrete prediction {concrete_predictions}"

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
    
    