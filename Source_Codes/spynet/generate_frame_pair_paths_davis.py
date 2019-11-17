import os
from glob import glob
from os.path import *

saveFileName = 'sequences_path.txt'
davisPath = '/home/zhang205/Github/Datasets/DAVIS'
seqsPath = join(davisPath, 'JPEGImages', '480p')
lines = []

for d in [join(seqsPath, f) for f in glob(join(seqsPath, '*/'))]:
    print('Processing sequence: ', d)

    # Gets frames in sequence
    seq_frames = sorted(glob(join(d, '*.jpg')))
    frame_pairs = [seq_frames[i] + ' ' + seq_frames[i+1] + '\n' for i in range(len(seq_frames) - 1)]
    lines = lines + frame_pairs

print(lines)

# Writes paths to file
with open(saveFileName, 'w') as f:
    f.writelines(lines)
