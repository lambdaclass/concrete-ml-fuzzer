import sys
import random
import atheris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr_sklearn
from concrete.ml.sklearn import LogisticRegression as lr_concrete

# Dataset to train
x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

# Split the data-set into a train and test set,
# each set is split into input and result.
input_train, _, result_train, _ = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Start the concrete-ml logistic regression model
concrete_model = lr_concrete()
# Train
concrete_model.fit(input_train, result_train)

# Compile FHE
concrete_model.compile(input_train)

# Start the sklearn logistic regression model
scikit_model = lr_sklearn()
# Train
scikit_model.fit(input_train, result_train)

#test = [[random.random() for _ in range(30)] for _ in range(2)]

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(30, 0.0, 1.0) for _ in range(2)]
    # Run the inference on encrypted inputs
    fhe_pred = concrete_model.predict(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_model.predict(data)
    assert((fhe_pred == prediction).all())

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
