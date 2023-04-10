import numpy as np
import sys
import atheris
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import PoissonRegressor as ConcretePoissonRegressor

df, _ = fetch_openml(
    data_id=41214, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
)
df = df.head(50000)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
train_data = df_train["DrivAge"].values.reshape(-1, 1)
test_data = np.sort(df_test["DrivAge"].values).reshape(-1, 1)
sklearn_pr = SklearnPoissonRegressor(max_iter=300)
sklearn_pr.fit(train_data, df_train["Frequency"], sample_weight=df_train["Exposure"])
concrete_pr = ConcretePoissonRegressor(n_bits=8)
concrete_pr.fit(train_data, df_train["Frequency"], sample_weight=df_train["Exposure"])
concrete_pr.compile(train_data)

def compare_models(input_bytes):
    fdp = atheris.FuzzedDataProvider(input_bytes)
    #data = [fdp.ConsumeFloatListInRange(5, -1.0, 1.0) for _ in range(100)]
    sklearn_predictions = sklearn_pr.predict(test_data)
    concrete_predictions = concrete_pr.predict(test_data, execute_in_fhe=True)
    assert((concrete_predictions == sklearn_predictions).all())

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
