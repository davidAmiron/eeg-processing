"""Utility for data analysis"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
