"""Process the results of a stationarity test"""

import sys
import pandas as pd
import numpy as np

significance_values = [0.05, 0.01, 0.005]
files = [
    'results/stationarity-full-result.csv',
    'results/stationarity-full-sra-result.csv',
    'results/stationarity-full-diff-result.csv',
    'results/stationarity-full-diff-sra-result.csv'
]

for filename in files:
    df = pd.read_csv(filename, index_col=0)
    total = np.prod(df.values.shape)
    print('Results for {}'.format(filename))
    for sig_value in significance_values:
        num_signif = np.count_nonzero(df.values <= sig_value)
        print('\tSignificance Value: {} -> {}/{} = {}'.format(sig_value, num_signif, total, num_signif/total))
    print()

