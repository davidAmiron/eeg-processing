import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
#%matplotlib inline

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.eval_measures import rmse, aic

# Import data (wage growth dataset)
# From here: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
#print(df.shape)
#print(df.tail())
#df = df.diff().dropna().diff().dropna() # Line to see what differenced data looks like

# Display data
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10, 6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)

    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
#plt.show()

# Run tests

# Automated Dickey-Fuller Test
def adf_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    result = adfuller(series, autolag='AIC')
    test_statistic = round(result[0], 4)
    pvalue = round(result[1], 4)
    n_lags = round(result[2], 4)
    n_obs = result[3]
    def adjust(val, length=6): return str(val).ljust(length)

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

for name, column in df.diff().dropna().diff().dropna().iteritems():
    adf_test(column, name=column.name)
    print()

# Granger test for causality
'''maxlag = 12
test = 'ssr_chi2test' # One of 'lrtest', 'params_ftest', 'ssr_chi2test', 'ssr_ftest'
def granger_causation_matrix(data, variables, test='ssr_chi2test', zero_diagonal=False, verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in df.columns:
        for row in df.index:
            if col == row and zero_diagonal:
                df.loc[row, col] = 0
                continue
            test_result = grangercausalitytests(data[[row, col]], maxlag=maxlag, verbose=False)
            """if col == 'rgnp' and row == 'pgnp':
                print(data[[row, col]])
                pprint(test_result[1][0]['ssr_chi2test'])
                sys.exit(0)"""
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {row}, X = {col}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[row, col] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

granger_df = granger_causation_matrix(df, variables=df.columns, zero_diagonal=True)
print(granger_df)
fig, ax = plt.figure(), plt.gca()
ax.imshow(granger_df, cmap='jet')
ax.set_xticks(list(range(len(granger_df.columns))))
ax.set_xticklabels(granger_df.columns)
ax.set_yticks(list(range(len(granger_df.index))))
ax.set_yticklabels(granger_df.index)
plt.show()

granger_df_binary = pd.DataFrame().reindex_like(granger_df)
granger_df_binary[granger_df < 0.01] = 1
granger_df_binary.fillna(0, inplace=True)
print(granger_df_binary)'''

# Split data into train and test
nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]
#print(df_test.shape)

# Difference dataframe to make it stationary (see article, am skipping some steps just to use the model)
df_train_differenced = df_train.diff().dropna().diff().dropna()

# Model with different orders
model = VAR(df_train_differenced)
for p in range(1, 10):
    result = model.fit(p)
    print('Lag Order: {}'.format(p))
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

# Choose model of order 4
model_fitted = model.fit(4)
print(model_fitted.summary())

# Forecast differenced data
lag_order = model_fitted.k_ar
forecast_input = df_train_differenced.values[-lag_order:]

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')

# Invert differenced data
def invert_difference(df_train, df_forecast, second_diff=False):
    """Undo differencing to get forecasting at original scale"""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results_all = invert_difference(df_train, df_forecast, second_diff=True)
df_results = df_results_all[[str(col) + '_forecast' for col in df_train.columns]]

# Plot forecasts
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=100, figsize=(10, 10))
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    ax.plot(df_test[col][-nobs:], label='Actual')
    ax.plot(df_results[col+'_forecast'], label='Forecast')
    #ax.autoscale(axis='x', tight=True)
    ax.legend()

    ax.set_title(col + ': Forecast v. Actuals')
    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()
