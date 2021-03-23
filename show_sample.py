"""Display a signal from an experiment, subject, and block, with or without differences"""

import sys
import os
import matplotlib.pyplot as plt

from utils import read_file_raw

if len(sys.argv) < 6:
    print("Usage: python show_sample.py [path/to/MURIBCI] [experiment] [subject] [block] [electrode] [differences=0]")
    print("electrode should be 'reference_0', 'reference_1', or 'signal_0' through 'signal_64'")
    sys.exit(1)

database_loc = sys.argv[1]
experiment = sys.argv[2]
subject = int(sys.argv[3])
block = int(sys.argv[4])
electrode = sys.argv[5]
if len(sys.argv) == 7:
    differences = sys.argv[6]

filename = 'CCexp{}_Subj_{:03d}_{:02d}.mat'.format(experiment, subject, block)
raw = read_file_raw(database_loc, filename)

sig_type = electrode.split('_')[0]
sig_type = sig_type[0].capitalize() + sig_type[1:]
if electrode.find('_') == -1:
    print("Invalid electrode. should be 'reference_0', 'reference_1', or 'signal_0' through 'signal_64'")
    sys.exit(1)
sig_num = int(electrode.split('_')[1])

data = raw[sig_type][0, :, sig_num]

plt.plot(data)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Subject {}, Block {}, Electrode {}'.format(subject, block, electrode))
plt.show()
