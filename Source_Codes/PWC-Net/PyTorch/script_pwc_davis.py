import sys
import cv2
import torch
import numpy as np
from os.path import *
import os
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
from glob import glob
from time import time
import models

"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def evalFramePair(im1_fn, im2_fn, flow_fn):

    im_all = [imread(img) for img in [im1_fn, im2_fn]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)
    for i in range(len(im_all)):
        im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i] / 255.0

        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        im_all[_i] = torch.from_numpy(im_all[_i])
        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])
        im_all[_i] = im_all[_i].float()

    im_all = torch.autograd.Variable(torch.cat(im_all, 1).cuda(), volatile=True)

    net = models.pwc_dc_net(pwc_model_fn)
    net = net.cuda()
    net.eval()
    
    # Benchmarks model eval time per image pair
    start = time()
    flo = net(im_all)
    time_elapsed = time() - start

    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2)  #
    u_ = cv2.resize(flo[:, :, 0], (W, H))
    v_ = cv2.resize(flo[:, :, 1], (W, H))
    u_ *= W / float(W_)
    v_ *= H / float(H_)
    flo = np.dstack((u_, v_))
    
    writeFlowFile(flow_fn, flo)
    return time_elapsed


DAVIS_path = '/home/zhang205/Github/Datasets/DAVIS';
seqs_path = join(DAVIS_path, 'JPEGImages', '480p')

if len(sys.argv) > 1:
    DAVIS_path = sys.argv[1]

save_path = join(DAVIS_path, 'OpticalFlow', '480p_PWC_Net')

if not os.path.exists(save_path):
    os.mkdir(save_path)

pwc_model_fn = './pwc_net.pth.tar';

n_flowmaps = 0.0
total_time = 0.0

for d in [join(seqs_path, f) for f in glob(join(seqs_path, '*/'))]:

    # Gets frames in sequence
    seq_frames = sorted(glob(join(d, '*.jpg')))

    # Creates list of frame pairs in sequence
    frame_pairs = [(seq_frames[i], seq_frames[i + 1]) for i in range(len(seq_frames) - 1)]

    # Creates directory for saving sequence
    save_dir = os.path.join(save_path, os.path.basename(d[:-1]))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Evaluates each frame pair in d
    for (img1_fn, img2_fn) in frame_pairs:
        flow_save_path = join(save_dir, os.path.basename(os.path.splitext(img1_fn)[0]) + '.flo')
        print('Processing {}'.format(flow_save_path))

        # Benchmarks evaluation time for frame pair
        time_elapsed = evalFramePair(img1_fn, img2_fn, flow_save_path)
        
        # counts n_flowmaps and total_time
        total_time += time_elapsed
        n_flowmaps += 1.0

print('Done!')
print('Average eval time: {}s per flowmap'.format(total_time / n_flowmaps))
