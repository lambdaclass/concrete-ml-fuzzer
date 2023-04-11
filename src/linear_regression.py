import sys
import atheris
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

# Regression train set
input_train, result_train = make_regression(n_samples=100, n_features=5, random_state=42)

concrete_model = ConcreteLinearRegression().fit(input_train, result_train)
concrete_model.compile(input_train)

scikit_model = SklearnLinearRegression().fit(input_train, result_train)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(5, -3.0, 3.0) for _ in range(100)]
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Compare both outputs
    assert((fhe_pred == prediction).all())

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
