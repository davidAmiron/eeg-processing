"""Display a signal from an experiment, subject, and block, with or without differences"""

import sys
import os
import matplotlib.pyplot as plt

from utils import read_file_raw

if len(sys.argv) < 6:
    print("Usage: python show_sample.py [path/to/MURIBCI] [experiment] [subject] [block] [electrode] [sub_ref_avg=false] [differences=0]")
    print("electrode should be 'reference_0', 'reference_1', or 'signal_0' through 'signal_63'")
    sys.exit(1)

database_loc = sys.argv[1]
experiment = sys.argv[2]
subject = int(sys.argv[3])
block = int(sys.argv[4])
electrode = sys.argv[5]

sub_ref_avg = 0
if len(sys.argv) >= 7:
    if sys.argv[6].lower() == 'true':
        sub_ref_avg = True
    elif sys.argv[6].lower() == 'false':
        sub_ref_avg = False
    else:
        print('sub_ref_avg should be \'True\' or \'False\'')
        sys.exit(1)

differences = 0
if len(sys.argv) >= 8:
    differences = int(sys.argv[7])

filename = 'CCexp{}_Subj_{:03d}_{:02d}.mat'.format(experiment, subject, block)
data = read_file_raw(database_loc, filename, sub_ref_avg=sub_ref_avg)

if electrode.find('_') == -1:
    print("Invalid electrode. should be 'reference_0', 'reference_1', or 'signal_0' through 'signal_63'")
    sys.exit(1)

sample_data = data[electrode]
for i in range(differences):
    print('Differencing')
    sample_data = sample_data.diff().dropna()
print(sample_data)

plt.plot(sample_data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Subject {}, Block {}, Electrode {}'.format(subject, block, electrode))
plt.show()
