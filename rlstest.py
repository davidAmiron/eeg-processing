import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg

from utils import *

# Script Arguments
if len(sys.argv) < 7:
    print('Usage: python rls.py [path/to/MURIBCI] [num_train] [num_test] [lags] [lookahead_distance] [lambda]')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])
p = int(sys.argv[4])
l = int(sys.argv[5])
lam = float(sys.argv[6])

fs = 2000

print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

data_train = data['2b']['001']['01'][['signal_37']][:num_train].values.T[0]
data_test = data['2b']['001']['01'][['signal_37']][num_train:num_train + num_test].values.T[0]

#lags = [1, 5, 10, 15, 20, 25, 30, 35, 40]
lags = [10, 20, 30, 40, 50]
#lambdas = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
lambdas = [0.8, 0.85, 0.9, 0.95, 0.99]
#lookaheads = np.arange(1, 50)
lookaheads = np.arange(1, 50)

"""results = {}
for lag in lags:
    results[lag] = {}
    for lamm in lambdas:
        results[lag][lamm] = {}
        results[lag][lamm]['train_mse'] = []
        results[lag][lamm]['train_mae'] = []
        results[lag][lamm]['test_mse'] = []
        results[lag][lamm]['test_mae'] = []
        print('Lag {}, lambda {}'.format(lag, lamm))
        for lookahead in lookaheads:
            rls = RLS(lag, lamm, 100, lookahead)
            data_train_pred, train_mse, train_mae = rls.fit(data_train)
            data_test_aug = np.concatenate([data_train[-(lookahead+lag):], data_test])
            test_pred, test_mse, test_mae = rls.test(data_test_aug)
            results[lag][lamm]['train_mse'].append(train_mse)
            results[lag][lamm]['train_mae'].append(train_mae)
            results[lag][lamm]['test_mse'].append(test_mse)
            results[lag][lamm]['test_mae'].append(test_mae)"""


rls = RLS(p, lam, 100, l)
data_train_pred, train_mse, train_mae = rls.fit(data_train)
#print('mse mae ', train_mse, train_mae)

# Data test with last l+p values of data_train for pred
data_test_aug = np.concatenate([data_train[-(l+p):], data_test])
test_pred, test_mse, test_mae = rls.test(data_test_aug)
print('Test MSE: {}'.format(test_mse))
print('Test MAE: {}'.format(test_mae))


lw = 1
plt.plot(data_train, label='True', linewidth=lw)
plt.plot(data_train_pred, label='Predicted', linewidth=lw)
plt.legend()
plt.title('Train')

plt.figure()
plt.plot(data_test_aug, label='True', linewidth=lw)
#plt.plot(test_pred[:l+p], label='From Train', linewidth=lw)
plt.plot(test_pred, label='Predicted', linewidth=lw)
plt.legend()
#plt.ylim((12945, 13025))
plt.title('Test')
plt.show()

"""for metric in ['mse', 'mae']:
    fig, axs = plt.subplots(2, len(lambdas))
    axs[0, 0].set_ylabel('Train {}'.format(metric.upper()))
    axs[1, 0].set_ylabel('Test {}'.format(metric.upper()))
    for i_lamm, lamm in enumerate(lambdas):
        axs[0, i_lamm].set_title('lambda {}'.format(lamm))
        for lag in lags:
            axs[0, i_lamm].plot(lookaheads, results[lag][lamm]['train_{}'.format(metric)], label='Lag {}'.format(lag))
            axs[1, i_lamm].plot(lookaheads, results[lag][lamm]['test_{}'.format(metric)], label='Lag {}'.format(lag))
        axs[0, i_lamm].legend()
        axs[1, i_lamm].legend()

    fig.suptitle('{} metrics for RLS Model'.format(metric.upper()))
"""
plt.show()
