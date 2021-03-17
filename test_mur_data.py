from os import listdir
from os.path import isfile, join
import scipy.io
from pprint import pprint
import matplotlib.pyplot as plt

from utils import *

'''mat = scipy.io.loadmat('MURIBCI/data/Exp2b/CCexp2b_Subj_001_01.mat')
print(type(mat))
print(mat['Signal'].shape)'''

# Parameters
num_signal_electrodes = 1       # Test run for ony first electrode
num_reference_electrodes = 2

# Get list of data files
path = 'MURIBCI/data/Exp2b'
exclude_files = ['Cap_coords_64.csv', 'README.txt', 'Protocol.png']
data_files = [f for f in listdir(path) if isfile(join(path, f)) and f not in exclude_files]

# Parse files into datastructure
print('Loading data...')
data = {}
for filename in data_files:
    filetype = filename.split('.')[-1]
    file_fields = filename[5:].split('.')[0].split('_')
    experiment = file_fields[0]
    subject = file_fields[2]
    block = file_fields[3]
    if filetype == 'mat':
        data.setdefault(experiment, {}).setdefault(subject, {})[block] = scipy.io.loadmat(join(path, filename))
    elif filetype == 'txt':
        # Fix above line if importing txt files cuz is setting, if txt file is read first will be overwritten
        pass
#print('Data loaded. Starting Augmented Dickey Fuller test...')

# Run stationarity test on data
print('Running stationarity tests')
for subject, subject_data in data['2b'].items():
    for block, block_data in subject_data.items():
        print('Processing subject {}, block {}'.format(subject, block))
        print()
        for i in range(num_reference_electrodes):
            print('Running ADF test for subject {}, block {}, reference electrode {}.'.format(subject, block, i))
            pvalue = adf_test(data['2b'][subject][block]['Reference'][0, :, i])
        for i in range(num_signal_electrodes):
            print('Running ADF test for subject {}, block {}, signal electrode {}.'.format(subject, block, i))
            pvalue = adf_test(data['2b'][subject][block]['Signal'][0, :, i])

#adf_test(a, name='test', print_results=True)
#plt.plot(a)
#plt.show()
