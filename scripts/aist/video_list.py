import os
import numpy as np

filenames = np.loadtxt('aist_plusplus_final/splits/all.txt', dtype=str)
ignore_filenames = np.loadtxt('aist_plusplus_final/ignore_list.txt', dtype=str)
filenames = np.array(sorted(list(set(filenames) - set(ignore_filenames))))
filename_dict = {}

for filename in filenames:
    strings = filename.split('_')
    ch = strings[-1]
    music = strings[-2]
    prefix = '_'.join(strings[:-2])

    if prefix in filename_dict:
        if ch in filename_dict[prefix]:
            filename_dict[prefix][ch].append(music)
        else:
            filename_dict[prefix][ch] = [music]
    else:
        filename_dict[prefix] = {ch: [music]}

new_filenames = []
for prefix in filename_dict:
    for ch in filename_dict[prefix]:
        strings = [prefix, filename_dict[prefix][ch][0], ch]
        string = '_'.join(strings)
        new_filenames.append(string)
