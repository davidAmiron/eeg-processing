import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime

from utils import *

# Script Arguments
if len(sys.argv) < 4:
    print('Usage: python test_mur_data.py [path/to/MURIBCI] [path/to/output_folder (without slash on end)] [partition]')
    print('run python -u if directing standard output to a file, to see prints as they are called')
    sys.exit(1)

database_loc = sys.argv[1]
output_folder = sys.argv[2]
partition = int(sys.argv[3])

start_time = datetime.now()

if partition == 0:
    subject = '001'
    #blocks = ['01', '02', '03']
    blocks = ['01']
elif partition == 1:
    subject = '001'
    blocks = ['04', '05', '06']
elif partition == 2:
    subject = '002'
    blocks = ['01', '02', '03']
elif partition == 3:
    subject = '002'
    blocks = ['04', '05', '06']
else:
    print('Invalid partition, must be 0, 1, 2, or 3')
    sys.exit(1)

# Parse files into datastructure
print('Loading data...')
data, columns, recordings_index = load_mur_data(database_loc, sub_ref_avg=True)
print('Data loaded')

"""
print(data['2b']['001']['01'][['signal_0', 'signal_1', 'signal_2']])
#a = grangercausalitytests(data['2b']['001']['01'][['signal_0', 'signal_1']], maxlag=12)
#print(a)
signal_cols = data['2b']['001']['01'].columns[2:]
a = granger_causation_matrix(data['2b']['001']['01'], signal_cols, 15, test='params_ftest', print_results=True)
print(a)
# See what happens when give test 3 values
"""

# Run granger causaity tests on data
print('Running granger causation tests')
signal_cols = data['2b']['001']['01'].columns[2:]
for block in blocks:
    print('Running for subject {}, block {}'.format(subject, block))
    #result = granger_causation_matrix(data['2b'][subject][block], signal_cols, 10, test='params_ftest',
    #                                  print_results=True, modified_granger=False)
    df_pvals, df_fstats = granger_causation_matrix(data['2b'][subject][block], signal_cols, 10, test='params_ftest',
                                      print_results=True, modified_granger=True)
    #print('Result:')
    #print(result)
    print('F Statistics:')
    print(df_fstats)
    print('p-values')
    print(df_pvals)
    #result.to_csv('{}/subj_{}_block_{}_causation.csv'.format(output_folder, subject, block))
    df_pvals.to_csv('{}/subj_{}_block_{}_causation_pvals.csv'.format(output_folder, subject, block))
    df_fstats.to_csv('{}/subj_{}_block_{}_causation_fstats.csv'.format(output_folder, subject, block))

"""modified_granger_test(data['2b']['001']['01'], 'signal_0', 'signal_27', 50, 0.1, 0.01,
                      var_calc_mode='all', difference=True)"""

run_time = datetime.now() - start_time
print('Time to run: {}'.format(run_time))
