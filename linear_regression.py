"""Run linear regression to predict one electrode from the others"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression

from utils import *

def main():
    num_subjects = 2
    num_blocks = 6

    if len(sys.argv) < 2:
        print('Usage: python linear_regression.py [database_loc]')
        sys.exit(1)

    database_loc = sys.argv[1]

    data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
    all_blocks = []
    for block_num in range(1, num_blocks):
        block_str = '{:02d}'.format(block_num)
        all_blocks.append(data['2b']['001'][block_str][columns[2:]])
    data_s1_train = pd.concat(all_blocks, ignore_index=True)
    data_s1_test = data['2b']['001']['06'][columns[2:]]

    x_cols = columns[3:]
    y_cols = columns[2]
    print('x_cols: {}'.format(x_cols))
    print('y_cols: {}'.format(y_cols))
    model = fit_model(data_s1_train, x_cols, y_cols)
    print(model.coef_)
    print()
    print(model.intercept_)

    print(data_s1_test[x_cols])
    prediction = model.predict(data_s1_test[x_cols])

    plt.plot(data_s1_test[y_cols], label='Truth')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()


    """b1 = data['2b']['001']['01']
    print(b1[['signal_0', 'signal_1']])

    x = b1[['signal_0', 'signal_1']]
    y = b1[['signal_2', 'signal_3']]
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    print(model.coef_)
    print()
    print(model.intercept_)"""

def fit_model(data, x_columns, y_columns):
    """Fit a linear model

    Args:
        x_columns (List): List of column names of explanatory variables
        y_columns (List): List of column names to predict

    Returns:
        The fitted linear model

    """
    model = LinearRegression(fit_intercept=True)
    X = data[x_columns]
    y = data[y_columns]
    model.fit(X, y)
    return model
    


def potato():
    print('potato boi')







if __name__ == '__main__':
    main()
