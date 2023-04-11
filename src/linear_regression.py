import sys
import atheris
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

# Dataset to train
input_train, result_train, _ = make_regression(n_samples=100, n_features=5, random_state=42)

# Start the concrete-ml linear regression model, train (unencrypted data) and quantize the weights.
concrete_model = ConcreteLinearRegression()
concrete_model.fit(input_train, result_train)

# Compile FHE
concrete_model.compile(input_train)

# Start the sklearn linear regression model
scikit_model = SklearnLinearRegression()
# Train
scikit_model.fit(input_train, result_train)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(5, -1.0, 1.0) for _ in range(100)]
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Compare both outputs
    assert((fhe_pred == prediction).all())

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
