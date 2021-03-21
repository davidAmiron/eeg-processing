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
num_signal_electrodes = 1
num_reference_electrodes = 2

# Parse files into datastructure
print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')
#print(data['2b']['002']['06']['signal_0'])
#sys.exit(0)
#print(recordings_index)

"""for i in range(num_reference_electrodes):
    df_cols.append(reference_index_str(i))
for i in range(num_signal_electrodes):
    df_cols.append(signal_index_str(i))"""


# Run stationarity test on data
print('Running stationarity tests')
df_pvalues = pd.DataFrame(index=recordings_index, columns=columns)
for subject, subject_data in data['2b'].items():
    for block, block_data in subject_data.items():
        print('Processing subject {}, block {}'.format(subject, block))
        for i in range(num_reference_electrodes):
            print('Running ADF test for subject {}, block {}, reference electrode {}.'.format(subject, block, i))
            df_pvalues.loc[subj_block_str(subject, block), reference_index_str(i)] = \
                    adf_test(data['2b'][subject][block][reference_index_str(i)].diff().dropna(),
                             name=subj_block_str(subject, block), print_results=True)
        for i in range(num_signal_electrodes):
            print('Running ADF test for subject {}, block {}, signal electrode {}.'.format(subject, block, i))
            df_pvalues.loc[subj_block_str(subject, block), signal_index_str(i)] = \
                    adf_test(data['2b'][subject][block][signal_index_str(i)].diff().dropna(),
                             name=subj_block_str(subject, block), print_results=True)

print('P Values:')
print(df_pvalues)

# Save p values
df_pvalues.to_csv(output_file)

run_time = datetime.now() - start_time
print('Time to run: {}'.format(run_time))

