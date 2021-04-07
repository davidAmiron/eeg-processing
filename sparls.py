import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg

from utils import *

# Script Arguments
if len(sys.argv) < 8:
    print('Usage: python sparls.py [path/to/MURIBCI] [num_train] [num_test] [lags_max] [lookahead_distance] [gamma] [alpha]')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])
max_lags = int(sys.argv[4])
l = int(sys.argv[5])
gamma = float(sys.argv[6])
alpha = float(sys.argv[7])

fs = 2000

print('Loading data...')
#data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

#data_train = data['2b']['001']['01'][['signal_37']][:num_train].values.T[0]
#data_test = data['2b']['001']['01'][['signal_37']][num_train:num_train + num_test].values.T[0]

sparls = SPARLS(2, 1)
print(np.arange(-10, 10))
print(sparls.soft_threshold(np.arange(-10, 10)))
