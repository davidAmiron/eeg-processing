import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import *

# Script Arguments
if len(sys.argv) < 6:
    print('Usage: python lookahead_ar.py [path/to/MURIBCI] [num_train] [num_test] [lags] [lookahead_distance]')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])
lags = int(sys.argv[4])
lookahead_distance = int(sys.argv[5])

fs = 2000

print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

data_train = data['2b']['001']['01'][['signal_37']][:num_train]
data_test = data['2b']['001']['01'][['signal_37']][num_train:num_train + num_test]

# Look at autocovariances
"""autocorr_row = [autocovariance(data_train_diff, i) for i in range(len(data_train_diff))]
plt.plot(autocorr_row, linewidth=1)
plt.xlabel('Autocorrelation index')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Differences')
plt.show()
sys.exit(0)"""

def train_and_pred_lar(data_train, data_test, lags, lookahead_distance, fs=2000, plot_coeffs=False, plot_sm_model=False):
    data_train_diff = data_train.diff().dropna()
    data_test_diff = data_test.diff().dropna()

    # Train my model
    print('Fitting...')
    fit = fit_lookahead_ar(data_train_diff, lookahead_distance, lags)
    print('Model fit: {}'.format(fit))
    if plot_coeffs:
        plt.figure()
        plt.plot(fit)
        plt.xlabel('Lag')
        plt.ylabel('Coefficient')
        plt.title('Fit Coefficients')
        plt.show()


    # Predict differences
    train_prediction_diff = [np.dot(fit, data_train_diff[i:i + lags][::-1])[0]
            for i in range(len(data_train_diff) - lags - lookahead_distance)]

    #sys.exit(0)
    train_prediction = [data_train.values[i + lags] + np.sum(train_prediction_diff[i:i + lookahead_distance])
            for i in range(lookahead_distance, len(data_train.values) - (lags + lookahead_distance))]

    #train_prediction = data_train.values[lags + lookahead_distance + 1:].T[0] + train_prediction_diff

    test_prediction_diff = [np.dot(fit, data_test_diff[i:i + lags][::-1])[0]
            for i in range(len(data_test_diff) - lags - lookahead_distance)]

    test_prediction = [data_test.values[i + lags] + np.sum(test_prediction_diff[i:i + lookahead_distance])
            for i in range(lookahead_distance, len(data_test.values) - (lags + lookahead_distance))]


    #test_prediction = data_test.values[lags + lookahead_distance + 1:].T[0] + test_prediction_diff

    # Train statsmodels model
    if lookahead_distance == 0 and plot_sm_model:
        sm_model = AutoReg(data_train_diff, lags=lags, trend='n', old_names=False)
        sm_fit = sm_model.fit()
        train_prediction_diff_sm = sm_fit.predict(start=lags).values
        sm_fit_params = sm_fit.params.values
        print(sm_fit_params)
        test_prediction_diff_sm = [np.dot(sm_fit_params, data_test_diff[i: i + lags][::-1])
                for i in range(len(data_test_diff) - lags)]

    return {
            'train_diff': data_train_diff,
            'test_diff': data_test_diff,
            'pred_train': train_prediction,
            'pred_train_diff': train_prediction_diff,
            'pred_test': test_prediction,
            'pred_test_diff': test_prediction_diff
           }

# CHECK WHAT HAPPENS IF FEED IN NOT PANDAS DATAFRAME (do I really need to do this?)
train_mse = {}
test_mse = {}
train_mae = {}
test_mae = {}

train_diff_mse = {}
test_diff_mse = {}
train_diff_mae = {}
test_diff_mae = {}
for lags in range(10, 51, 10):
#for lags in [10, 20]:
    train_mse[lags] = []
    test_mse[lags] = []
    train_mae[lags] = []
    test_mae[lags] = []

    train_diff_mse[lags] = []
    test_diff_mse[lags] = []
    train_diff_mae[lags] = []
    test_diff_mae[lags] = []
    for lookahead_distance in range(31):
    #for lookahead_distance in [0, 1]:
        print('Getting results for lag {}, lookahead {}'.format(lags, lookahead_distance))
        results = train_and_pred_lar(data_train, data_test, lags, lookahead_distance)
        train_mse[lags].append(mean_squared_error(data_train[lags+2*lookahead_distance:], results['pred_train']))
        train_mae[lags].append(mean_absolute_error(data_train[lags+2*lookahead_distance:], results['pred_train']))

        test_mse[lags].append(mean_squared_error(data_test[lags+2*lookahead_distance:], results['pred_test']))
        test_mae[lags].append(mean_absolute_error(data_test[lags+2*lookahead_distance:], results['pred_test']))

        train_diff_mse[lags].append(mean_squared_error(results['train_diff'][lags+lookahead_distance:],
            results['pred_train_diff']))
        train_diff_mae[lags].append(mean_absolute_error(results['train_diff'][lags+lookahead_distance:],
            results['pred_train_diff']))

        test_diff_mse[lags].append(mean_squared_error(results['test_diff'][lags+lookahead_distance:],
            results['pred_test_diff']))
        test_diff_mae[lags].append(mean_absolute_error(results['test_diff'][lags+lookahead_distance:],
            results['pred_test_diff']))

# x values for plots to place predicted further over because of lag and lookahead
"""train_true_xvals = list(range(len(data_train)))
train_pred_xvals = list(range(lags + 2 * lookahead_distance + 0, len(data_train)))
test_true_xvals = list(range(len(data_test)))
test_pred_xvals = list(range(lags + 2 * lookahead_distance + 0, len(data_test)))

train_true_xvals_diff = list(range(len(result['train_diff'])))
train_pred_xvals_diff = list(range(lags + lookahead_distance, len(result['train_diff'])))
test_true_xvals_diff = list(range(len(result['test_diff'])))
test_pred_xvals_diff = list(range(lags + lookahead_distance, len(result['test_diff'])))

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
linewidth = 1

axs[0, 0].plot(train_true_xvals, data_train, linewidth=linewidth, label='True')
axs[0, 0].plot(train_pred_xvals, result['pred_train'], linewidth=linewidth, label='Predicted')
axs[0, 0].legend()
axs[0, 0].set_title('Actual Train')

axs[0, 1].plot(test_true_xvals, data_test, linewidth=linewidth, label='True')
axs[0, 1].plot(test_pred_xvals, result['pred_test'], linewidth=linewidth, label='Predicted')
axs[0, 1].legend()
axs[0, 1].set_title('Actual Test')

axs[1, 0].plot(train_true_xvals_diff, result['train_diff'], linewidth=linewidth, label='True')
axs[1, 0].plot(train_pred_xvals_diff, result['pred_train_diff'], linewidth=linewidth, label='Predicted')
#if lookahead_distance == 0 and plot_sm_model:
#    axs[1, 0].plot(np.array(train_pred_xvals_diff) + 1, train_prediction_diff_sm,
#            linewidth=linewidth, label='Predicted (statsmodels)')
axs[1, 0].legend()
axs[1, 0].set_title('Differenced Train')

axs[1, 1].plot(test_true_xvals_diff, result['test_diff'], linewidth=linewidth, label='True')
axs[1, 1].plot(test_pred_xvals_diff, result['pred_test_diff'], linewidth=linewidth, label='Predicted')
#if lookahead_distance == 0 and plot_sm_model:
#    axs[1, 1].plot(test_pred_xvals_diff, test_prediction_diff_sm,
#            linewidth=linewidth, label='Predicted (statsmodels)')
axs[1, 1].legend()
axs[1, 1].set_title('Differenced Test')

fig.suptitle('Lookahead AR\n\nLags = {}, Lookahead = {}\nTrain: {} timesteps ({}s)\nTest: {} timesteps ({}s)'
        .format(lags, lookahead_distance, num_train, num_train/fs, num_test, num_test/fs))

plt.subplots_adjust(top=0.85)
plt.show()"""

fig, axs = plt.subplots(2, 4, figsize=(12, 9))
axs[0, 0].set_title('Train MSE')
axs[0, 0].set_ylabel('MSE')
for lag, errors in train_mse.items():
    axs[0, 0].plot(errors, label='Lag {}'.format(lag))

axs[1, 0].set_title('Train MAE')
axs[1, 0].set_xlabel('Lookahead')
axs[1, 0].set_ylabel('MAE')
for lag, errors in train_mae.items():
    axs[1, 0].plot(errors, label='Lag {}'.format(lag))

axs[0, 1].set_title('Test MSE')
for lag, errors in test_mse.items():
    axs[0, 1].plot(errors, label='Lag {}'.format(lag))

axs[1, 1].set_title('Test MAE')
axs[1, 1].set_xlabel('Lookahead')
for lag, errors in test_mae.items():
    axs[1, 1].plot(errors, label='Lag {}'.format(lag))

axs[0, 2].set_title('Train Difference MSE')
for lag, errors in train_diff_mse.items():
    axs[0, 2].plot(errors, label='Lag {}'.format(lag))

axs[1, 2].set_title('Train Differences MAE')
axs[1, 2].set_xlabel('Lookahead')
for lag, errors in train_diff_mae.items():
    axs[1, 2].plot(errors, label='Lag {}'.format(lag))

axs[0, 3].set_title('Test Differences MSE')
for lag, errors in test_diff_mse.items():
    axs[0, 3].plot(errors, label='Lag {}'.format(lag))

axs[1, 3].set_title('Test Differences MAE')
axs[1, 3].set_xlabel('Lookahead')
for lag, errors in test_diff_mae.items():
    axs[1, 3].plot(errors, label='Lag {}'.format(lag))

for ax in axs.flatten():
    ax.legend()

fig.suptitle('Lookahead AR\n\nTrain: {} timesteps ({}s)\nTest: {} timesteps ({}s)'
        .format(num_train, num_train/fs, num_test, num_test/fs))
plt.subplots_adjust(top=0.82)
#plt.tight_layout()
plt.show()
# Prime decomposition lookahead autoregression: A special case of generalized advantage estimation
