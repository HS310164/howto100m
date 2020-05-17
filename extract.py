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
parser.add_argument('--input2d_dir', type=str, help='input 2d file directory')
parser.add_argument('--input3d_dir', type=str, help='input 3d file directory')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path')
args = parser.parse_args()

class ExDataset(data.Dataset):
    def __init__(self, file2d_path, file3d_path):
        super(ExDataset, self).__init__()
        self.files2d = sorted(glob.glob(os.path.join(file2d_path, '*.npy')))
        self.files3d = sorted(glob.glob(os.path.join(file3d_path, '*.npy')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.basename(self.files2d[index])
        if not file_name == os.path.basename(self.files3d[index]):
            raise ValueError('not same file name')
        file_name = os.path.splitext(file_name)[0]
        vi2d = np.load(self.files2d[index])[0]
        vi3d = np.load(self.files3d[index])[0]
        vi = np.concatenate([vi2d, vi3d], axis=1)
        return vi, file_name

    
def main():
    dataset = ExDataset(args.input2d_dir, args.input3d_dir)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    net = Net(
        video_dim=4096,
        embd_dim=6144,
        we_dim=300,
        max_words=20
        )
    net.load_checkpoint(args.checkpoint_path)
    net.eval()
    net.cuda()
    for inputs, file_name in dl:
        output = []
        with th.no_grad():
            for dim in inputs:
                dim = dim.reshape([1, -1])
                print(dim.shape)
                video = net(dim.cuda())
                output.append(video.to('cpu').data.numpy())
            outputs = np.array(output)
            print('finish {}'.format(file_name))
            out_path = os.path.join(args.output_dir, file_name + '.npy')
            np.save(out_path, outputs)

if __name__ == "__main__":
    main()




