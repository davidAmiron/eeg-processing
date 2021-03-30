import os
import sys
import pandas as pd
import numpy as np

num_subjects = 2
num_blocks = 6

if len(sys.argv) < 2:
    print('Usage: python process_causality.py [results_folder]')
    sys.exit(1)

results_folder = sys.argv[1]

for subject in range(1, num_subjects + 1):
    for block in range(1, num_blocks + 1):
        filename = os.path.join(results_folder, 'subj_{:03d}_block_{:02d}_causation.csv'.format(subject, block))
        df = pd.read_csv(filename, index_col=0)
        print('Subject {}, Block {}'.format(subject, block))
        print(type(df.values))
        print(np.count_nonzero(df.values))
        print()
