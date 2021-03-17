from os import listdir
import os.path
import sys
from datetime import datetime
import scipy.io
from pprint import pprint
import matplotlib.pyplot as plt

from utils import *

'''mat = scipy.io.loadmat('MURIBCI/data/Exp2b/CCexp2b_Subj_001_01.mat')
print(type(mat))
print(mat['Signal'].shape)'''

start_time = datetime.now()

# Script Arguments
if len(sys.argv) < 3:
    print('Usage: python test_mur_data.py [path/to/MURIBCI] [path/to/output_file.csv]')
    print('run python -u if directing standard output to a file, to see prints as they are called')
    sys.exit(1)

database_loc = sys.argv[1]
output_file = sys.argv[2]

# Parameters
num_signal_electrodes = 1       # Test run for ony first electrode
num_reference_electrodes = 2

# Get list of data files
data_path = os.path.join(database_loc, 'data/Exp2b')
exclude_files = ['Cap_coords_64.csv', 'README.txt', 'Protocol.png']
data_files = [f for f in listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f not in exclude_files]
data_files.sort() # Put in alphanumerical order

# Functions for names of rows and columns of output
def subj_block_str(subject, block):
    return 'subj_{}_block_{}'.format(subject, block)
def signal_index_str(index):
    return 'signal_{}'.format(index)
def reference_index_str(index):
    return 'reference_{}'.format(index)

# Parse files into datastructure
print('Loading data...')
data = {}
df_index = []
df_cols = []
for filename in data_files:
    filetype = filename.split('.')[-1]
    if filetype == 'mat':
        file_fields = filename[5:].split('.')[0].split('_')
        experiment = file_fields[0]
        subject = file_fields[2]
        block = file_fields[3]
        data.setdefault(experiment, {}).setdefault(subject, {})[block] = scipy.io.loadmat(os.path.join(data_path, filename))
        df_index.append(subj_block_str(subject, block))

    elif filetype == 'txt':
        # Ignoring txt files for now
        pass

for i in range(num_reference_electrodes):
    df_cols.append(reference_index_str(i))
for i in range(num_signal_electrodes):
    df_cols.append(signal_index_str(i))

print('Data loaded')

# Run stationarity test on data
print('Running stationarity tests')
df_pvalues = pd.DataFrame(index=df_index, columns=df_cols)
for subject, subject_data in data['2b'].items():
    for block, block_data in subject_data.items():
        print('Processing subject {}, block {}'.format(subject, block))
        for i in range(num_reference_electrodes):
            print('Running ADF test for subject {}, block {}, reference electrode {}.'.format(subject, block, i))
            df_pvalues.loc[subj_block_str(subject, block), reference_index_str(i)] = \
                    adf_test(data['2b'][subject][block]['Reference'][0, :, i], print_results=True)
        for i in range(num_signal_electrodes):
            print('Running ADF test for subject {}, block {}, signal electrode {}.'.format(subject, block, i))
            df_pvalues.loc[subj_block_str(subject, block), signal_index_str(i)] = \
                    adf_test(data['2b'][subject][block]['Signal'][0, :, i], print_results=True)

print('P Values:')
print(df_pvalues)

# Save p values
df_pvalues.to_csv(output_file)


run_time = datetime.now() - start_time
print('Time to run: {}'.format(run_time))

#adf_test(a, name='test', print_results=True)
#plt.plot(a)
#plt.show()
