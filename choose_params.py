"""Choose hyperparameters for all models and generate graphs displaying reasons for choice"""

import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import *

# Script Arguments
if len(sys.argv) < 2:
    print('Usage: python choose_params.py [path/to/MURIBCI]')
    sys.exit(1)

database_loc = sys.argv[1]

num_train = 5000
num_validate = 1000
num_test = 1000

num_subjects = 1
num_blocks = 1 #6

max_lags = 100
lookahead = 10

method = 'sparls'

stat_method = 'mae'
def summary_stat(x, y):
    if stat_method == 'mae':
        return mean_absolute_error(x, y)
    elif stat_method == 'mse':
        return mean_squared_error(x, y)

print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

# Lookahead AR model
if method == 'lar':
    print('Lookahead AR Parameter Selection')

    lar_stats_total = []
    lags = np.arange(1, max_lags+1)
    for subject in range(1, num_subjects + 1):
        for block in range(1, num_blocks + 1):
            sub_str = '{:03d}'.format(subject)
            blk_str = '{:02d}'.format(block)
            print('Subject {}, block {}'.format(sub_str, blk_str))
            for signal in columns[2:]:
                print('Processing signal {}'.format(signal))
                data_train = data['2b'][sub_str][blk_str][[signal]][:num_train]
                data_validate = data['2b'][sub_str][blk_str][[signal]][num_train:num_train + num_validate]

                data_train_diff = data_train.diff().dropna()
                data_validate_diff = data_validate.diff().dropna()

                lag_stats = []
                for lag in lags:
                    #print('Lag {}'.format(lag))
                    w = fit_lookahead_ar(data_train_diff, lookahead, lag)

                    # Predict differences
                    #train_prediction_diff = [np.dot(w, data_train_diff[i:i + lag][::-1])[0]
                    #        for i in range(len(data_train_diff) - lag - lookahead)]

                    validate_prediction_diff = [np.dot(w, data_validate_diff[i:i + lag][::-1])[0]
                            for i in range(len(data_validate_diff) - lag - lookahead)]

                    # Reconstruct signals
                    #train_prediction = [data_train.values[i + lag] + np.sum(train_prediction_diff[i:i + lookahead])
                    #        for i in range(lookahead, len(data_train.values) - (lag + lookahead))]

                    validate_prediction = [data_validate.values[i + lag] + np.sum(validate_prediction_diff[i:i + lookahead])
                            for i in range(lookahead, len(data_validate.values) - (lag + lookahead))]

                    # Calculate stats
                    validate_stat = summary_stat(data_validate[lag+2*lookahead:], validate_prediction)
                    #print('Test {}: {}'.format(stat_method, validate_stat))

                    lag_stats.append(validate_stat)
                lar_stats_total.append(lag_stats)


                #plt.plot(lags, stats)
                #plt.show()
                #print()

    lar_stats_total = np.array(lar_stats_total)
    channel_means = np.mean(lar_stats_total, axis=0)
    lag_index = np.argmin(channel_means)
    lag_chosen = lags[lag_index]
    print('LAR Channel Means:\n{}'.format(channel_means))
    print('LAR Lag chosen: {}'.format(lag_chosen))
    

# RLS
if method == 'rls':
    print('RLS Parameter Selection')

    rls_stats_total = []
    lags = np.arange(1, max_lags+1)
    lambdas = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
    #lambdas = [0.8, 0.85, 0.9, 0.975]
    for subject in range(1, num_subjects + 1):
        for block in range(1, num_blocks + 1):
            sub_str = '{:03d}'.format(subject)
            blk_str = '{:02d}'.format(block)
            print('Subject {}, block {}'.format(sub_str, blk_str))
            for signal in columns[2:]:
                print('Processing signal {}'.format(signal))
                data_train = data['2b'][sub_str][blk_str][[signal]][:num_train].values.T[0]
                data_validate = data['2b'][sub_str][blk_str][[signal]][num_train:num_train + num_validate].values.T[0]


                lag_stats = []
                for lag in lags:
                    print('Lag {}'.format(lag))
                    lambda_stats = []
                    for lam in lambdas:
                        rls = RLS(lag, lam, 100, lookahead)
                        rls.fit(data_train, nostats=True, nopred=True)
                        data_validate_aug = np.concatenate([data_train[-(lookahead+lag):], data_validate])
                        validate_pred = rls.test(data_validate_aug, nostats=True)
                        validate_stat = summary_stat(data_validate_aug[lag+lookahead:], validate_pred[lag+lookahead:])
                        lambda_stats.append(validate_stat)
                    lag_stats.append(lambda_stats)
                rls_stats_total.append(lag_stats)

    rls_stats_total = np.array(rls_stats_total)
    rls_channel_means = np.mean(rls_stats_total, axis=0)
    lag_i, lambda_i = np.unravel_index(np.argmin(rls_channel_means), rls_channel_means.shape)
    rls_lag_chosen = lags[lag_i]
    rls_lambda_chosen = lambdas[lambda_i]
    print('Lag chosen: {}'.format(rls_lag_chosen))
    print('Lambda chosen: {}'.format(rls_lambda_chosen))
    with open('rls_validation_stats.npy', 'wb') as fh:
        np.save(fh, rls_channel_means)
    print(rls_channel_means)

if method == 'sparls':
    print('SPARLS Parameter Selection')

    sparls_stats_total = []
    #lambdas = [0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
    lambdas = [0.8, 0.9, 0.975]
    for subject in range(1, num_subjects + 1):
        for block in range(1, num_blocks + 1):
            sub_str = '{:03d}'.format(subject)
            blk_str = '{:02d}'.format(block)
            print('Subject {}, block {}'.format(sub_str, blk_str))
            for signal in columns[2:5]:
                print('Processing signal {}'.format(signal))
                data_train = data['2b'][sub_str][blk_str][[signal]][:num_train].values.T[0]
                data_validate = data['2b'][sub_str][blk_str][[signal]][num_train:num_train + num_validate].values.T[0]
                data_train -= np.mean(data_train)
                data_validate -= np.mean(data_validate)

                lambda_stats = []
                for lam in lambdas:
                    print('Lambda: {}'.format(lam))
                    gamma = 0.005
                    alpha = 0.2
                    sigma = 5000
                    sparls = SPARLS(gamma, alpha, sigma, lam, max_lags, lookahead, 50)
                    sparls.fit(data_train, nopred=True, nostats=True)
                    validate_pred = sparls.test(data_validate, nostats=True)
                    validate_stat = summary_stat(data_validate[max_lags+lookahead:], validate_pred[max_lags+lookahead:])
                    lambda_stats.append(validate_stat)
                sparls_stats_total.append(lambda_stats)

    sparls_stats_total = np.array(sparls_stats_total)
    sparls_channel_means = np.mean(sparls_stats_total, axis=0)
    lambda_i = np.argmin(sparls_channel_means)
    sparls_lambda_chosen = lambdas[lambda_i]
    print('Lambda chosen: {}'.format(sparls_lambda_chosen))
    with open('sparls_validation_stats.npy', 'wb') as fh:
        np.save(fh, sparls_channel_means)
    print(sparls_channel_means)
