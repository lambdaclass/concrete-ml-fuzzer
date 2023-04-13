import sys 
import numpy
import atheris
from sklearn.datasets import make_blobs
from concrete.ml.sklearn.rf import RandomForestClassifier

train_x, train_y = make_blobs(n_samples=1000, n_features=10, centers=100,
    random_state=42)
concrete_model, sklearn_model = RandomForestClassifier(n_bits=12, n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=42).fit_benchmark(train_x, train_y)
concrete_model.compile(train_x)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(10, -13.0, 13.0) for _ in range(100)]
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True).flatten()
    # Get scikit prediction
    sk_pred = sklearn_model.predict(data)

    # Get the mean absolute percentage error(mape) to make sure the accuaracy is around 99%
    mape = numpy.mean(numpy.abs((sk_pred - fhe_pred) / sk_pred))

    # Compare outputs
    assert(mape < 1.0)

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
