import sys
import atheris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression

# Dataset to train
x, y = make_classification(n_samples=100, class_sep=2, n_features=5, random_state=42)

# Split the data-set into a train and test set,
# each set is split into input and result.
input_train, _, result_train, _ = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Start the concrete-ml logistic regression model, train (unencrypted data) and quantize the weights.
concrete_model = ConcreteLogisticRegression(n_bits=12)
concrete_model.fit(input_train, result_train)

# Compile FHE
concrete_model.compile(input_train)

# Start the sklearn logistic regression model
scikit_model = SklearnLogisticRegression()
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
