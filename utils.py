"""Utility for data analysis"""

from os import listdir
import os.path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

from statsmodels.tsa.api import VAR
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

def granger_causation_matrix(data, variables, test='ssr_chi2test', zero_diagonal=False, print_results=False):
    """Check Granger Causality of all possible combinations of the Time series.

    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    Args:
        data: pandas dataframe containing the time series variables
        variables: list containing names of the time series variables to analyze
        test: what test to use ('lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest')
        zero_diagonal: True to set all diagonal entries to zero
        print_results: True to print results

    Returns:
        Pandas Dataframe, matrix of causations. Each row is a response variable, and
        each column is a predictor (denoted by names ending in '_y' or '_x'

    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in df.columns:
        for row in df.index:
            if zero_diagonal and col == row:
                df.loc[row, col] = 0
                continue
            test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if print_results:
                print(f'Y = {row}, X = {col}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[row, col] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

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
    assert(num_reference_electrodes <= 2)
    assert(num_signal_electrodes <= 64)

    # Get list of data files
    data_path = os.path.join(database_loc, 'data/Exp2b')
    exclude_files = ['Cap_coords_64.csv', 'README.txt', 'Protocol.png']
    data_files = [f for f in listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f not in exclude_files]
    data_files.sort() # Put in alphanumerical order

    data = {}
    columns = []
    ref_columns = []
    recordings_index = []

    num_reference_electrodes = 2
    num_signal_electrodes = 64

    # Create columns
    for i in range(num_reference_electrodes):
        columns.append(reference_index_str(i))
        ref_columns.append(reference_index_str(i))
    for i in range(num_signal_electrodes):
        columns.append(signal_index_str(i))

    # Read data
    for filename in data_files:
        filetype = filename.split('.')[-1]
        if filetype == 'mat':
            file_fields = filename[5:].split('.')[0].split('_')
            experiment = file_fields[0]
            subject = file_fields[2]
            block = file_fields[3]
            df = read_file_raw(database_loc, filename, sub_ref_avg=sub_ref_avg)

            data.setdefault(experiment, {}).setdefault(subject, {})[block] = df
            recordings_index.append(subj_block_str(subject, block))

        elif filetype == 'txt':
            # Ignoring txt files for now
            pass

    return data, columns, recordings_index
