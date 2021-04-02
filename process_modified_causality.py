import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

num_subjects = 2
num_blocks = 6

if len(sys.argv) < 2:
    print('Usage: python process_causality.py [results_folder]')
    sys.exit(1)

results_folder = sys.argv[1]

subject = 1
block = 1
#fstats_filename = 'subj_{:03d}_block_{:02d}_causation_fstats.csv'.format(subject, block)
#pvals_filename = 'subj_{:03d}_block_{:02d}_causation_pvals.csv'.format(subject, block)
fstats_filename = 'subj_{:03d}_block_{:02d}_causation_all_fstats.csv'.format(subject, block)
pvals_filename = 'subj_{:03d}_block_{:02d}_causation_all_pvals.csv'.format(subject, block)
fstats_file = os.path.join(results_folder, fstats_filename)
pvals_file = os.path.join(results_folder, pvals_filename)

df_fstats = pd.read_csv(fstats_file, index_col=0)
df_pvals = pd.read_csv(pvals_file, index_col=0)
print(df_pvals)
plt.figure()
axp = sns.heatmap(df_pvals)
axp.set_title('p-values')

plt.figure()
axf = sns.heatmap(df_fstats, vmin=-6, vmax=16)
axf.set_title('f statistics')
plt.show()

"""for subject in range(1, num_subjects + 1):
    for block in range(1, num_blocks + 1):
        filename = os.path.join(results_folder, 'subj_{:03d}_block_{:02d}_causation.csv'.format(subject, block))
        df = pd.read_csv(filename, index_col=0)
        print('Subject {}, Block {}'.format(subject, block))
        print(type(df.values))
        print(np.count_nonzero(df.values))
        print()"""
