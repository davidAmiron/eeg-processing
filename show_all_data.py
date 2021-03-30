import matplotlib.pyplot as plt

from utils import *

data, columns, recordings_index = load_mur_data('./MURIBCI', sub_ref_avg=True)

fig, axs = plt.subplots(6, 2)

for s, (subject, subject_data) in enumerate(data['2b'].items()):
    for b, (block, block_data) in enumerate(subject_data.items()):
        axs[b, s].plot(block)

plt.show()
