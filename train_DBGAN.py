from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

import model
import numpy as np
import pylib
import tensorboardX
import torch
import torchlib
import os
from tqdm import tqdm

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# command line arguments
parser = argparse.ArgumentParser()
# model
parser.add_argument('--z_dim', dest='z_dim', type=int, default=121)
# training
parser.add_argument('--epoch', dest='epoch', type=int, default=100)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
parser.add_argument('--d_learning_rate', dest='d_learning_rate', type=float, default=0.0002)
parser.add_argument('--g_learning_rate', dest='g_learning_rate', type=float, default=0.001)
parser.add_argument('--loss_mode', dest='loss_mode', choices=['gan', 'lsgan', 'wgan', 'hinge_v1', 'hinge_v2'], default='gan')
parser.add_argument('--gp_mode', dest='gp_mode', choices=['none', 'dragan', 'wgan-gp'], default='none')
parser.add_argument('--gp_coef', dest='gp_coef', type=float, default=1.0)
parser.add_argument('--norm', dest='norm', choices=['none', 'batch_norm', 'instance_norm'], default='none')
parser.add_argument('--weight_norm', dest='weight_norm', choices=['none', 'spectral_norm', 'weight_norm'], default='spectral_norm')
# others
parser.add_argument('--experiment_name', dest='experiment_name', default='CGAN_default')
parser.add_argument('--data_path', dest='data_path', default='./data')
# parse arguments
args = parser.parse_args()
# model
z_dim = args.z_dim
# training
epoch = args.epoch
batch_size = args.batch_size
d_learning_rate = args.d_learning_rate
g_learning_rate = args.g_learning_rate
loss_mode = args.loss_mode
gp_mode = args.gp_mode
gp_coef = args.gp_coef
norm = args.norm
weight_norm = args.weight_norm
# ohters
experiment_name = args.experiment_name
data_path = args.data_path

# ==============================================================================
# =                                   setting                                  =
# ==============================================================================

# data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        files = sorted(os.listdir(self.path))
        assert len(files) % 4 == 0
        self.len = len(files) // 4
        self.labels_list = []
        self.read_iat_list = []
        self.read_segment_nid_list = []
        self.write_iat_list = []
        self.write_buffer_size_list = []
        self.write_segment_nid_list = []
        for i in range(0, self.len):
            label_array = np.load(os.path.join(self.path, f'labels_{i}.npy')).astype('float32')
            self.labels_list.append(label_array)
            
            write_iat_array = np.load(
                os.path.join(self.path, f'write_iat_{i}.npy'))
            self.write_iat_list.append(write_iat_array)

            write_buffer_size_array = np.load(
                os.path.join(self.path, f'write_buffer_size_{i}.npy'))
            self.write_buffer_size_list.append(write_buffer_size_array)

            write_segment_nid_array = np.load(
                os.path.join(path, f'write_segment_nid_{i}.npy'))
            self.write_segment_nid_list.append(write_segment_nid_array)

    def __getitem__(self, index):
        return self.labels_list[index], torch.tensor([
            self.write_iat_list[index], self.write_buffer_size_list[index],
            self.write_segment_nid_list[index]
        ])

    def __len__(self):
        return self.len

train_loader = torch.utils.data.DataLoader(
    dataset = MyDataset(data_path),
    batch_size=batch_size,
    shuffle=True
)

# model
c_dim = 7
device = torch.device("cuda")
D = model.DiscriminatorDBGAN(x_dim=3, c_dim=c_dim).to(device)
G = model.GeneratorDBGAN(z_dim=z_dim, c_dim=c_dim).to(device)

# gan loss function
d_loss_fn, g_loss_fn = model.get_losses_fn(loss_mode)
loss = torch.nn.functional.mse_loss

# optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

# ==============================================================================
# =                                    train                                   =
# ==============================================================================

ckpt_dir = './output/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    start_ep = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_ep = 0

# writer
writer = tensorboardX.SummaryWriter('./output/%s/summaries' % experiment_name)

for ep in tqdm(range(start_ep, epoch)):
    for i, (c_dense, x) in enumerate(train_loader):
        step = ep * len(train_loader) + i + 1
        D.train()
        G.train()

        # train D
        x = x.to(device)
        c = torch.tensor(c_dense).to(device)
        z = torch.randn(c.size(0), z_dim).to(device)
#         print(c)

        x_f = G(z, c).detach()
        x_gan_logit = D(x)
        x_f_gan_logit = D(x_f)
        
        d_x_f_gan_loss = loss(x_f_gan_logit, torch.zeros(x_gan_logit.size()).to(device))
        d_x_gan_loss = loss(x_gan_logit, c)

#         d_x_gan_loss, d_x_f_gan_loss = d_loss_fn(x_gan_logit, x_f_gan_logit)
        gp = model.gradient_penalty(D, x, x_f, mode=gp_mode)
        d_loss = d_x_gan_loss + d_x_f_gan_loss + gp * gp_coef

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/d_gan_loss', (d_x_gan_loss + d_x_f_gan_loss).data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        # train G
        z = torch.randn(c.size(0), z_dim).to(device)

        x_f = G(z, c)
        x_f_gan_logit = D(x_f)

        g_gan_loss = loss(x_f_gan_logit, c)
        g_loss = g_gan_loss

        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        writer.add_scalar('G/g_gan_loss', g_gan_loss.data.cpu().numpy(), global_step=step)

    torchlib.save_checkpoint({'epoch': ep + 1,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'd_optimizer': d_optimizer.state_dict(),
                              'g_optimizer': g_optimizer.state_dict()},
                             '%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1),
                             max_keep=2)
