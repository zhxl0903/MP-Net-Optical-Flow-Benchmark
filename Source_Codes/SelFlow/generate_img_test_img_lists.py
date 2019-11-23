import os
from glob import glob
from os.path import *

saveFilePath = './img_list'
davisPath = './DAVIS'

seqsPath = join(davisPath, 'JPEGImages', '480p')

for d in glob(join(seqsPath, '*/')):
    print('Processing sequence: ', d)

    # Gets frames in sequence
    seq_frames = sorted(list(map(lambda img: basename(img), glob(join(d, '*.jpg')))))

    # Gets first frame triplet
    frame_triplets = [seq_frames[0] + ' ' + seq_frames[0] + ' ' + seq_frames[1] + ' ' + '00000' + '\n']

    # Appends remaining frame triplets
    frame_triplets += [seq_frames[i - 1] + ' ' + seq_frames[i] + ' ' + seq_frames[i + 1] + ' ' + str(i).zfill(5) + '\n'
                       for i
                       in
                       range(1, len(seq_frames) - 1)]

    print(frame_triplets)

    # Writes paths to file with name
    with open(join(saveFilePath, os.path.basename(d[:-1]) + '.txt'), 'w') as f:
        f.writelines(frame_triplets)
