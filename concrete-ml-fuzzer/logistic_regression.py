from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr_sklearn
from concrete.ml.sklearn import LogisticRegression as lr_concrete

# Dataset to train
x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

# Split the data-set into a train and test set,
# each set is split into input and result.
input_train, input_test, result_train, _ = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Start the concrete-ml logistic regression model
concrete_model = lr_concrete()
# Train
concrete_model.fit(input_train, result_train)

# Compile FHE
concrete_model.compile(input_train)

# Run the inference on encrypted inputs
fhe_pred = concrete_model.predict(input_test, execute_in_fhe=True)

# Start the sklearn logistic regression model
scikit_model = lr_sklearn()
# Train
scikit_model.fit(input_train, result_train)

# Get scikit prediction
prediction = scikit_model.predict(input_test)

assert((fhe_pred == prediction).all())
