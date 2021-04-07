import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg

from utils import *

# Script Arguments
if len(sys.argv) < 4:
    print('Usage: python compare.py [path/to/MURIBCI] [num_train] [num_test] ')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])

# Parameters
fs = 2000

# Load data
print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

# Create training and test sets
signal_columns = columns[2:]

data_train = [data['2b']['001']['01'][[col]][:num_train].values.T[0] for col in signal_columns]
data_test = [data['2b']['001']['01'][[col]][num_train:num_train + num_test].values.T[0] for col in signal_columns]

for i, (train_set, test_set) in enumerate(zip(data_train, data_test)):
    print(i, train_set, test_set)
