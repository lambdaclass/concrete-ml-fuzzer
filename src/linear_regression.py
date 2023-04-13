import sys
import numpy
import atheris
from sklearn.datasets import make_regression
from concrete.ml.sklearn import LinearRegression
# Regression train set
input_train, result_train = make_regression(n_samples=100, n_features=5, random_state=42)

concrete_model, scikit_model = LinearRegression(n_bits=12).fit_benchmark(input_train, result_train)
concrete_model.compile(input_train)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(5, -1.5, 4.0) for _ in range(100)]
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True).flatten()
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Get the mean absolute percentage error(mape) to make sure the accuaracy is around 99%
    mape = numpy.mean(numpy.abs((prediction - fhe_pred) / prediction))

    # Compare outputs
    assert(mape < 1.0)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
