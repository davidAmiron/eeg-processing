import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg

from utils import *

# Script Arguments
if len(sys.argv) < 8:
    print('Usage: python sparls.py [path/to/MURIBCI] [num_train] [num_test] [lags_max] [lookahead_distance] [gamma] [alpha]')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])
max_lags = int(sys.argv[4])
l = int(sys.argv[5])
gamma = float(sys.argv[6])
alpha = float(sys.argv[7])

fs = 2000

print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

data_train = data['2b']['001']['01'][['signal_43']][:num_train].values.T[0]
data_test = data['2b']['001']['01'][['signal_43']][num_train:num_train + num_test].values.T[0]

# Subtract mean
if True:
    data_train -= np.mean(data_train)
    data_test -= np.mean(data_test)

gamma = 0.005
alpha = 0.2
sigma = 1000
sparls = SPARLS(gamma, alpha, sigma, 0.95, 100, 10, 50)
data_train_pred, train_mse, train_mae = sparls.fit(data_train)
data_test_pred, test_mse, test_mae = sparls.test(data_test)

print('Train MSE: {}\n      MAE: {}'.format(train_mse, train_mae))
print('Test MSE: {}\n    MAE: {}'.format(test_mse, test_mae))


lw = 1
plt.plot(data_train, label='True', linewidth=lw)
plt.plot(data_train_pred, label='Predicted', linewidth=lw)
plt.title('Train')
plt.legend()

plt.figure()
plt.plot(data_test, label='True', linewidth=lw)
plt.plot(data_test_pred, label='Predicted', linewidth=lw)
plt.title('Test')
plt.legend()

plt.show()
