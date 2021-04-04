import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set()
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import *

# Script Arguments
if len(sys.argv) < 7:
    print('Usage: python rls.py [path/to/MURIBCI] [num_train] [num_test] [lags] [lookahead_distance] [lambda]')
    sys.exit(1)

database_loc = sys.argv[1]
num_train = int(sys.argv[2])
num_test = int(sys.argv[3])
p = int(sys.argv[4])
l = int(sys.argv[5])
lam = float(sys.argv[6])

fs = 2000

print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

data_train = data['2b']['001']['01'][['signal_37']][:num_train].values.T[0]
data_test = data['2b']['001']['01'][['signal_37']][num_train:num_train + num_test].values.T[0]

rls = RLS(p, lam, 1)

# x is feature series, d is the predicting series
# Put zeros of length p for initialization
# The zeros in front of d are unnecessary but made the indices work out nicer
x = np.concatenate([np.zeros(p), data_train[:-l]])
d = np.concatenate([np.zeros(p), data_train[l:]])
#d = data_train[l:]
N = len(x)

# Train
d_pred = np.zeros_like(d)
for n in range(p, N-1):
    print()
    w_n = rls.update(x[n-p:n+1][::-1], d[n])
    print('x[n]  d[n]', x[n], d[n])
    #print('Weights:\n{}'.format(w_n))
    #pred = rls.predict(x[n-p:n+1][::-1])
    pred = rls.predict(x[n-p+1:n+2][::-1])
    print('True ', d[n])
    print('Pred ', pred)
    d_pred[n+1] = pred

weights = rls.w

# Test
print('Test')

# Method 1 - Predict over lag using weights and retraining
#test_pred = np.zeros_like(data_test)
#test_pred[:p] = np.nan
"""data_test_aug = np.concatenate([data_train[:-(l+p)], data_test])    # Data test with last l+p values of data_train for pred
test_pred = np.concatenate([data_train[:-(l+p)], np.zeros_like(data_test)])
for n in range(p+l, len(data_test_aug)):
    test_pred[n] = rls.predict(test_pred[n-p-l:n-l+1])
    rls.update(test_pred[n-p-l+1:n-l+2], test_pred[n])"""



lw = 1
plt.plot(d[p:], label='True', linewidth=lw)
plt.plot(d_pred[p:], label='Predicted', linewidth=lw)
plt.legend()
plt.ylim((12950, 13025))
plt.show()

"""plt.plot(data_test, label='True', linewidth=lw)
plt.plot(test_pred[:l+p], label='From Train', linewidth=lw)
plt.plot(test_pred[l+p:], label='Predicted', linewidth=lw)
plt.legend()
plt.show()"""
