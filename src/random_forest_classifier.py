import sys 
import atheris
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from concrete.ml.sklearn.rf import RandomForestClassifier as ConcreteRandomForest

train_x, train_y = make_blobs(n_samples=1000, n_features=10, centers=100,
    random_state=42)

sklearn_model = SklearnRandomForest(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=42).fit(train_x, train_y)

concrete_model = ConcreteRandomForest(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=42).fit(train_x, train_y)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(10, -13.0, 13.0) for _ in range(100)]
    # Run the inference, encryption and decryption is done in the background
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)

    # Compare both outputs
    assert((fhe_pred == prediction).all())
