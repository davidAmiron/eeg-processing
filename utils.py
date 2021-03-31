"""Utility for data analysis"""

from os import listdir
import os.path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.stats

from statsmodels.tsa.api import VAR, AutoReg
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Automated Dickey-Fuller Test
def adf_test(series, signif=0.05, name='', print_results=False):
    """Perform ADFuller test for stationarity

    Args:
        series: The time series data to analyze
        signif: The significance level
        name: The name for printing results
        print_results: True to print all results

    Returns:
        pvalue for test


    """
    result = adfuller(series, autolag='AIC')
    test_statistic = round(result[0], 4)
    pvalue = round(result[1], 4)
    n_lags = round(result[2], 4)
    n_obs = result[3]
    def adjust(val, length=6): return str(val).ljust(length)

    if print_results:
        print('Augmented Dickey-Fuller Test on "{}"'.format(name), '\n', '-'*47)
        print('Null Hypothesis: Data has unit root. Non-Stationary')
        print('Significance Level: {}'.format(signif))
        print('Test Statistic: {}'.format(test_statistic))
        print('Number of Lags Chosen: {}'.format(n_lags))

        for key, val in result[4].items():
            print('Critical value {} = {}'.format(adjust(key), round(val, 3)))

        if pvalue <= signif:
            print(' => P-Value = {}. Rejecting Null Hypotheses.'.format(pvalue))
            print(' => Series is Stationary')
        else:
            print(' => P-Value = {}. Weak evidence to reject the Null Hypothesis'.format(pvalue))
            print(' => Series is Non-Stationary.')

    return pvalue

def granger_causation_matrix(data, variables, maxlag, test='ssr_chi2test', zero_diagonal=False,
                             print_results=False, modified_granger=False):
    """Check Granger Causality of all possible combinations of the Time series.

    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    Args:
        data: pandas dataframe containing the time series variables
        variables: list containing names of the time series variables to analyze
        maxlag: The maximum lag to go to
        test: what test to use ('lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest')
        zero_diagonal: True to set all diagonal entries to zero
        print_results: True to print results
        modified_granger: True to use modified granger causation

    Returns:
        Pandas Dataframe, matrix of causations. Each row is a response variable, and
        each column is a predictor (denoted by names ending in '_y' or '_x'). The pvalue
        reported is the minimum of the different lags up to max_lag

    Notes:
        Diagonal p-valeus are zeros, even when not processed with modified granger

    """
    df_pvals = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    if modified_granger:
        df_fstats = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in df_pvals.columns:
        for row in df_pvals.index:
            if zero_diagonal and col == row:
                df_pvals.loc[row, col] = 0
                continue
            if modified_granger:
                if row == col:  # Bug if they are equal, and don't actually care about those results
                    continue
                f_stat, p_value = modified_granger_test(data, row, col, 100, 0.1, 0.01,
                                                        var_calc_mode='point', difference=True)
                print(f_stat, p_value)
                df_pvals.loc[row, col] = p_value
                df_fstats.loc[row, col] = f_stat

            else:
                test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=print_results)
                p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
                if print_results:
                    print(f'Y = {row}, X = {col}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df_pvals.loc[row, col] = min_p_value

    df_pvals.columns = [var + '_x' for var in variables]
    df_pvals.index = [var + '_y' for var in variables]
    df_fstats.columns = [var + '_x' for var in variables]
    df_fstats.index = [var + '_y' for var in variables]
    if modified_granger:
        return df_pvals, df_fstats
    else:
        return df_pvals

def modified_granger_test(data_all, variable_caused, variable_causing, num_models,
                          train_time, pred_ahead_time, var_calc_mode='point', lags=10, fs=2000, difference=False):
    """Modified granger test between two variables

    In calculation of the F statistic, instead of using RSS, use sum of squares of
    prediction error on unseen data. These values will come from fitting multiple
    models and forecasting pred_ahead_time seconds into the future.

    Args:
        data_all (pd.DataFrame): The dataframe of data to use
        variable_caused (string): The column of data for the variable being caused
        variable_causing (string): The column of data for the variable causing the other
        num_models (int): The number of models to train and predictions to make
        train_time (float): The length of time of data for training each model (seconds)
        pred_ahead_time (float): The time into the future to use for prediction for F statistic (seconds)
        lags (int): Number of lags in models
        fs (int): Sampling frequency of data
        difference (bool): True to difference the data
        var_calc_mode (string): This determines how a model contributes to the total sum of squared prediction errors.
                                'point' - Only contribute the squared error of the forecast pred_ahead_time seconds
                                          into the future
                                'all'   - Add the squared error of prediction error from all of forecast up to
                                          pred_ahead_time seconds

    Notes:
        Models are spaced out as much as possible along data
    """

    if var_calc_mode not in ['point', 'all']:
        raise ValueError('var_calc_mode must be \'point\' or \'all\', got {}'.format(var_calc_mode))
        
    train_timesteps = int(train_time * fs)
    pred_timesteps = int(pred_ahead_time * fs)

    #caused = data[[variable_caused]]
    #causing = data[[variable_causing]]
    data = data_all[[variable_caused, variable_causing]]
    data_full = data
    data_restricted = data[[variable_caused]]

    col_caused_index = np.where(data.columns.values == variable_caused)[0][0]

    # Difference data
    if difference:
        #caused = caused.diff().dropna()
        #causing = causing.diff().dropna()
        #data = data.diff().dropna()
        data_full = data_full.diff().dropna()
        data_restricted = data_restricted.diff().dropna()


    # Make sure enough samples are available
    num_samples = data.shape[0]
    num_samples_req = train_timesteps + pred_timesteps + num_models - 1
    if num_samples_req > num_samples:
        raise ValueError('Number of samples required ({}) is greater than number of samples available ({}).'.format(
            num_samples_req, num_samples)
            )

    num_pred_available = num_samples - (train_timesteps + pred_timesteps)
    pred_spacing = int(num_pred_available / num_models)

    squared_errors_sum_full = 0
    squared_errors_sum_restricted = 0
    for i in range(0, num_pred_available, pred_spacing):

        # Create train and test data
        #print('i: {}, i + train_timesteps: {}'.format(i, i + train_timesteps))
        train_full = data_full[i:i+train_timesteps]
        train_restricted = data_restricted[i:i+train_timesteps]
        
        forecast_input_full = train_full[-lags:]
        forecast_input_restricted = train_restricted[-lags:]

        test_value = data[[variable_caused]].values[i + train_timesteps + pred_timesteps - 1][0]
        test_values_all = data[[variable_caused]].values[i + train_timesteps: i + train_timesteps + pred_timesteps].T[0]

        # Train models
        model_full = VAR(train_full.values)
        model_restricted = AutoReg(train_restricted.values, lags, old_names=False)

        model_full_fitted = model_full.fit(lags)
        model_restricted_fit = model_restricted.fit()

        # Forecast
        pred_full = model_full_fitted.forecast(y=forecast_input_full.values, steps=pred_timesteps)[:, col_caused_index]
        pred_restricted = model_restricted.predict(model_restricted_fit.params,
                train_timesteps, train_timesteps + pred_timesteps - 1)

        # Un-difference values
        if difference:
            pred_full = data[[variable_caused]].values[i + train_timesteps - 1] + pred_full.cumsum()
            pred_restricted = data[[variable_caused]].values[i + train_timesteps - 1] + pred_restricted.cumsum()
            #sys.exit(0)

        if var_calc_mode == 'point':
            squared_errors_sum_full += (test_value - pred_full[-1]) ** 2
            squared_errors_sum_restricted += (test_value - pred_restricted[-1]) ** 2
        elif var_calc_mode == 'all':
            squared_errors_sum_full += np.sum(np.square(pred_full - test_values_all))
            squared_errors_sum_restricted += np.sum(np.square(pred_restricted - test_values_all))
        
        """print('Full: {}\nRestricted: {}\nFull is less: \t\t\t\t{}'.format(squared_errors_sum_full,
                                                                squared_errors_sum_restricted,
                                                                squared_errors_sum_full < squared_errors_sum_restricted))"""

        """print(pred_full)
        print(pred_restricted)
        plt.plot(pred_full, label='Full')
        plt.plot(pred_restricted, label='Restricted')
        print(data_restricted.values[i + train_timesteps: i + train_timesteps + pred_timesteps])
        plt.plot(data[[variable_caused]].values[i + train_timesteps: i + train_timesteps + pred_timesteps], label='True')
        plt.legend()
        plt.show()"""
        #sys.exit(0)

    # Calculate F-statistic
    n = train_timesteps     # Number of datapoints for parameter estimation
    param_count_full = np.prod(model_full_fitted.params.shape) / 2      # Devide by two because params are for predicting two values
    param_count_restricted = len(model_restricted_fit.params)
    #print('full: {}'.format(param_count_full))
    #print('restricted: {}'.format(param_count_restricted))
    #print('full sum: {}\nrestricted sum: {}'.format(squared_errors_sum_full, squared_errors_sum_restricted))
    f_statistic = ((n - param_count_full) / (param_count_full - param_count_restricted)) \
                    * ((squared_errors_sum_restricted - squared_errors_sum_full) / squared_errors_sum_full)
    p_value = 1 - scipy.stats.f.cdf(f_statistic,
                                    param_count_full - param_count_restricted,
                                    n - param_count_full)
    return f_statistic, p_value






# Functions for names of rows and columns of output
def subj_block_str(subject, block):
    return 'subj_{}_block_{}'.format(subject, block)
def signal_index_str(index):
    return 'signal_{}'.format(index)
def reference_index_str(index):
    return 'reference_{}'.format(index)

def read_file_raw(database_loc, filename, sub_ref_avg):
    """Read a file in the database

    Args:
        database_loc (str): Location of MURIBCI folder
        filename (str): The file to read
        sub_ref_avg (bool): True to average the reference electrodes and subtract them from the signal
    """
    raw_read = scipy.io.loadmat(os.path.join(database_loc, 'data/Exp2b', filename))
    if sub_ref_avg:
        ref_avg = raw_read['Reference'][0].mean(axis=1)
        raw_read['Signal'][0] -= np.expand_dims(ref_avg, 1)

    num_reference_electrodes = 2
    num_signal_electrodes = 64

    columns = []
    for i in range(num_reference_electrodes):
        columns.append(reference_index_str(i))
    for i in range(num_signal_electrodes):
        columns.append(signal_index_str(i))

    all_data = np.hstack((raw_read['Reference'][0], raw_read['Signal'][0]))
    df = pd.DataFrame(all_data, columns=columns)
    return df

def load_mur_data(database_loc, sub_ref_avg=False):
    """Read in MURIBCI database

    Args:
        database_loc (str): Location of MURIBCI folder
        sub_ref_avg (bool): True to average the reference electrodes and subtract them from the signal

    Returns:
        A tuple containing (data, columns, recordings_index)
        data: dictionary indexed with [experiment][subject][block] -> pandas Dataframe of data
        columns: the columns (reference and signals)
        recordings_index: List of strings, one for each subject-block pair (1 recording)
        
    """

    # Get list of data files
    data_path = os.path.join(database_loc, 'data/Exp2b')
    exclude_files = ['Cap_coords_64.csv', 'README.txt', 'Protocol.png']
    data_files = [f for f in listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f not in exclude_files]
    data_files.sort() # Put in alphanumerical order

    data = {}
    recordings_index = []

    # Read data
    for filename in data_files:
        filetype = filename.split('.')[-1]
        if filetype == 'mat':
            file_fields = filename[5:].split('.')[0].split('_')
            experiment = file_fields[0]
            subject = file_fields[2]
            block = file_fields[3]
            df = read_file_raw(database_loc, filename, sub_ref_avg=sub_ref_avg)
            columns = df.columns

            data.setdefault(experiment, {}).setdefault(subject, {})[block] = df
            recordings_index.append(subj_block_str(subject, block))

        elif filetype == 'txt':
            # Ignoring txt files for now
            pass

    return data, columns, recordings_index
