# concrete-ml-fuzzer
A differential fuzzer to compare [concrete-ml](https://docs.zama.ai/concrete-ml) against [scikit-learn](https://scikit-learn.org)

## Dependencies
- scikit-learn
- concrete-ml
- atheris

These can be installed with:
```sh
pip install concrete-ml atheris
```
Installing concrete-ml with pip will install `scikit-learn` 

## Run the fuzzer

```sh
cd src
python3 logistic_regression.py
```
