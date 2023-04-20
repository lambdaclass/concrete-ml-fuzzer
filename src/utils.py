import atheris
import numpy as np
from sklearn.base import is_classifier
from sklearn.datasets import load_iris, make_regression
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)


def _classification_training() -> (np.array, np.array, dict):
    """
    Create a classification training set
    """
    training_x, training_y = load_iris(return_X_y=True)
    data_info = {
        "dimensions": 4,
        "min_feature": min(min(minimum) for minimum in training_x),
        "max_feature": max(max(maximum) for maximum in training_x),
    }

    return (training_x, training_y, data_info)

def _regression_training() -> (np.array, np.array, dict):
    """
    Create a regression training set
    """
    training_x, training_y = make_regression(n_samples=100, n_features=5, random_state=42)
    data_info = {
        "dimensions": 5,
        "min_feature": min(min(minimum) for minimum in training_x),
        "max_feature": max(max(maximum) for maximum in training_x),
    }

    return (training_x, training_y, data_info)

def _positive_regression_training() -> (np.array, np.array, dict, bool):
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
    data = data.head(1000)

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
        
    data_info = {
        "dimensions": 66,
        "min_feature": 0,
        "max_feature": 2000,
    }
    
    train_data, test_data, x_train_data, x_test_data = train_test_split(
        data,
        x,
        test_size=0.2,
        random_state=0,
    )
    
    gamma_mask_train = train_data["ClaimAmount"] > 0
    
    return (x_train_data, train_data, data_info, gamma_mask_train)


def consume_bytes(input_bytes: bytes, data_info, n_samples=1000, margin=0.1) ->  np.array:
    """
    Consume bytes and return a testing dataset for the models
    input_bytes: bytes generated by atheris
    data_info: dictionary provided by initialize_models; contains information to build the random datasets
    n_samples: the size of the dataset
    margin: margin to expand the dataset range beyond the range of the training set. Recomended to use small values
    """
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [
            fdp.ConsumeFloatListInRange(
                data_info["dimensions"], 
                data_info["min_feature"] - margin, 
                data_info["max_feature"] + margin
            ) for _ in range(n_samples)
           ]
    return data

def mean_absolute_percentage_error(y_sklearn, y_FHE) -> float:
    """
    Calculate the mean absolute percentage error to determine how much the FHE model deviates from the sklearn model.
    Values fall in the (0, 100) range, lower values represent less deviation.
    """
    
    # Compute accuracy for each possible representation
    score = np.abs((y_sklearn - y_FHE) / y_sklearn)

    return np.mean(score)


def initialize_models(ModelClass, params={"n_bits":12}):
    """
    Initialize concrete and sklearn models. Automatically load iris or diabetes dataset
    """
    
    X, y, data_info = _classification_training() if is_classifier(ModelClass) else _regression_training()
    
    # The n_bits parameter represents the number of bits used for the model quantization. 
    # Bigger numbers achieve a higher accuracy at the cost of speed. The maximum possible value (v1.0.0) is 12. 
    model = ModelClass(**params)
    concrete_model, sklearn_model = model.fit_benchmark(X, y)
    concrete_model.compile(X)
    return concrete_model, sklearn_model, data_info

def initialize_glm_models(concreteModelClass, scikitModelClass, type, params={"n_bits":12}):
    """
    Initialize concrete and sklearn models. Automatically load iris or diabetes dataset
    """
    
    x_train_data, train_data, data_info, gamma_mask_train = _positive_regression_training()
    
    fit_glm_parameters = { 
        "poisson": {
            "X": x_train_data,
            "y": train_data["Frequency"],
            "sample_weight": train_data["Exposure"],
        },
        "gamma": {
            "X": x_train_data[gamma_mask_train],
            "y": train_data[gamma_mask_train]["AvgClaimAmount"],
            "sample_weight": train_data[gamma_mask_train]["ClaimNb"],
        },
        "tweedie": {
            "X": x_train_data,
            "y": train_data["PurePremium"],
            "sample_weight": train_data["Exposure"],
        },
    }
    
    fit_parameters = (fit_glm_parameters["poisson"]
                  if type == "Poisson"
                  else fit_glm_parameters["gamma"]
                  if type == "Gamma"
                  else fit_glm_parameters["tweedie"])
    
    concrete_model = concreteModelClass(**params)
    sklearn_model = scikitModelClass()
    
    
    # Fit models
    concrete_model = concrete_model.fit(**fit_parameters) 
    sklearn_model = sklearn_model.fit(**fit_parameters)
    
    concrete_model.compile(x_train_data)
    return concrete_model, sklearn_model, data_info

def initialize_neural_models(ModelClass):
    """
    Initialize concrete and sklearn models. Automatically load iris or diabetes dataset
    """
    print(is_classifier(ModelClass))
    X, y, data_info = _classification_training() if is_classifier(ModelClass) else _regression_training()
    y = y.reshape(-1, 1)
    
    params = {
    "module__n_layers": 2,
    "module__n_a_bits": 3,
    "module__n_accum_bits": 8,
    "module__n_outputs": 3,
    "module__input_dim": X.shape[1],
    "module__activation_function": nn.Sigmoid,
    "max_epochs": 100,
    "verbose": 0,
    "n_bits":12,
}
    
    model = ModelClass(**params)
    concrete_model, sklearn_model = model.fit_benchmark(X.astype(np.float32), y.astype(np.float32))
    concrete_model.compile(X)
    return concrete_model, sklearn_model, data_info
