from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the training data
train_data = pd.read_csv('data/continuous/diabetes_data-original-train.csv')
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']

# Load the test data
test_data = pd.read_csv('data/continuous/diabetes_data-original-test.csv')
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as ConstKernel

# # Define the kernel function
kernel = ConstKernel(1.0) + ConstKernel(1.0) + RBF(10)

# # Create and train the GaussianProcessRegressor
# # optomize the hyperparameters using the gaussian process regressor
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gpr.fit(X_train, y_train)

# # Evaluate the GaussianProcessRegressor
# # Compute the covariance matrix
covariance_matrix = gpr.kernel_(X_train)
print(covariance_matrix)

# get the scores for the test data
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss
from scipy.special import kl_div
from sklearn.preprocessing import LabelBinarizer

# Predict the labels
y_train_pred = np.array(gpr.predict(X_train, return_std=False))
y_test_pred = np.array(gpr.predict(X_test, return_std=False))

# Compute the balanced accuracy
bal_acc_test = balanced_accuracy_score(y_test, y_test_pred.round())

# Compute the F1 score
f1_test = f1_score(y_test, y_test_pred.round())

# Compute the AUC
auc_test = roc_auc_score(y_test, y_test_pred)

# Compute the Brier score
brier_test = brier_score_loss(y_test, y_test_pred.round())

# Compute the KL divergence
lb = LabelBinarizer()
lb.fit(y_test)
y_test_lb = lb.transform(y_test)
y_test_pred_lb = lb.transform(y_test_pred.round())

def kl_divergence(Y_true, Y_prob):
        P = np.asarray(Y_true)+0.00001 # constant to avoid NAN in KL divergence
        Q = np.asarray(Y_prob)+0.00001 # constant to avoid NAN in KL divergence
        return np.sum(P*np.log(P/Q))
        
kl_test = kl_divergence(y_test_lb, y_test_pred_lb)

print(f'Test Balanced Accuracy: {bal_acc_test}')
print(f'Test F1 Score: {f1_test}')
print(f'Test AUC: {auc_test}')
print(f'Test Brier Score: {brier_test}')
print(f'Test KL Divergence: {kl_test}')