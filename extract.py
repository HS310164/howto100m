from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pickle
import glob
import argparse

import numpy as np
import torch as th
import torch.utils.data as data
from torch.utils.data import DataLoader

from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='input file directory')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path')
args = parser.parse_args()

class ExDataset(data.Dataset):
    def __init__(self, file_path):
        super(ExDataset, self).__init__()
        self.files = sorted(glob.glob(os.path.join(file_path, '*.npy')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.basename(self.files[index])
        file_name = os.path.splitext(file_name)[0]
        vi = np.load(self.files[index])
        return vi, file_name

    
def main():
    dataset = ExDataset(args.input_dir)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    net = Net()
    net.load_checkpoint(args.checkpoint_path)
    net.eval()
    net.cuda()
    for inputs, file_name in dl:
        output = []
        with th.no_grad():
            for dim in inputs:
                video = net(dim.cuda())
                output.append(video.to('cpu').data.numpy())
            outputs = np.array(output)
            print('finish {}'.format(file_name))
            out_path = os.path.join(args.output_dir, file_name + '.npy')
            np.save(out_path, outputs)

if __name__ == "__main__":
    main()




