import sys
import atheris
from random import uniform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import NeuralNetClassifier as ConcreteNN

train_input, train_result = make_classification(n_samples=100, class_sep=2, n_features=5, random_state=42)

concrete_model = ConcreteNN().fit(train_input, train_result)

concrete_model.compile()

data = [ [ uniform(-5.0, 5.0) for _ in range(5)] for _ in range(20) ]

pred = concrete_model.predict(data)
