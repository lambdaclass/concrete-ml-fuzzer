import sys
import random
import atheris
import concrete.tensorflow as ct
import tensorflow as tf
from tenseal import CKKSContext
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr_sklearn
from concrete.ml.sklearn import LogisticRegression as lr_concrete
    
def scikit_prediction(input_train, result_train, data):
    
    # Start the sklearn logistic regression model
    scikit_model = lr_sklearn()
    # Train
    scikit_model.fit(input_train, result_train)
    
    # convert the model to its homomorphic equivalent
    context = CKKSContext()
    context.generate_galois_keys()
    lr_encrypted = scikit_model.coef_.tolist()[0]
    for i in range(len(lr_encrypted)):
        lr_encrypted[i] = context.encrypt(lr_encrypted[i])
        
    # predict using the homomorphic model
    X_encrypted = []
    for i in range(data.shape[0]):
        X_encrypted.append(context.encrypt(data[i].tolist()))
    
    y_pred_encrypted = []
    for i in range(data.shape[0]):
        y_pred_encrypted.append(context.decrypt(context.multiply(lr_encrypted, X_encrypted[i])))

    y_pred = np.round(np.array(y_pred_encrypted))
    return y_pred


def concrete_prediction(input_train, result_train):
    # Start the concrete-ml logistic regression model
    concrete_model = lr_concrete()
    # Train
    concrete_model.fit(input_train, result_train)

    # Compile FHE
    concrete_model.compile(input_train)
    
    # Convert the model to its homomorphic equivalent
    homo_model = ct.keras_to_homo(model, encryption_params=c.DEFAULT_ENCRYPTION_PARAMETERS)

    # Encrypt the input data
    encrypted_data = ct.convert_to_tensor(data, homo_model)

    # Use the homomorphic model to make predictions
    encrypted_y = homo_model(encrypted_data)
    y_pred = ct.convert_to_numpy(encrypted_y)

    # Decrypt the predictions
    decrypted_y_pred = c.decrypt(c.DEFAULT_PRIVATE_KEY, y_pred)
    return decrypted_y_pred

def compare_models(input_bytes):
    # Dataset to train
    x, y = make_classification(n_samples=100, class_sep=2, n_features=30, random_state=42)

    # Split the data-set into a train and test set,
    # each set is split into input and result.
    input_train, _, result_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    fdp = atheris.FuzzedDataProvider(input_bytes)
    data = [fdp.ConsumeFloatListInRange(30, 0.0, 1.0) for _ in range(2)]
    # Run the inference on encrypted inputs
    fhe_pred = concrete_prediction(data, execute_in_fhe=True)
    # Get scikit prediction
    prediction = scikit_prediction(input_train, result_train)
    assert((fhe_pred == prediction).all())

atheris.Setup(sys.argv, compare_models)
atheris.Fuzz()
