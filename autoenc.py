# this import is completely unnessesray but on my machine there is a weird bug
# and this is the work around
import cv2
import h5py
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *

h5f = h5py.File('./inputs.h5', 'r') 
A = np.array(h5f['genotypes'])

A[A > 0.0] = 1.0
A[A < 0.0] = -1.0
first_row = A[0]

# F L U F F
print ("some statistics on fraction of non-zeros, it is abysmal ")
n_nonzero = np.count_nonzero(A)
print (n_nonzero / A.size)
print ("total matrix shape ", A.shape)

# S T U F F
N, M = A.shape

def gen_data():
  rand_row = random.randint(0, N-1)
  return A[rand_row]

def one_hot(d):
  def _one_hot(stuf):
    # negative
    if stuf == -1.0:
      return [1.0, 0.0]
    # positive
    if stuf == 1.0:
      return [0.0, 1.0]
    # unknown
    return [0.0, 0.0]

  ret = [_one_hot(dd) for dd in d]
  return np.array(ret)

def gen_data_batch(batch=30):
  ret = [one_hot(gen_data()) for _ in range(batch)]
  ret = np.array(ret)
  return to_torch(np.array(ret))

def all_batch():
  ret = [one_hot(d) for d in A]
  ret = np.array(ret)
  return to_torch(np.array(ret))

class AutoNet(nn.Module):

  def __init__(self):
    super(AutoNet, self).__init__()
    # for encding
    # 6 input image channel, 6 output channels, 2x2 square convolution
    self.enc_fc1 = nn.Linear(M*2, 50)
    self.enc_fc2 = nn.Linear(50, 6)
    self.dec_fc1 = nn.Linear(6, 50)
    self.dec_fc2 = nn.Linear(50, M*2)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


  def enc(self, x):
    x = x.view(-1, M*2)
    x = F.relu(self.enc_fc1(x))
    x = F.sigmoid(self.enc_fc2(x))
    return x

  def dec(self, x):
    x = self.dec_fc1(x)
    x = self.dec_fc2(x)
    x = x.view(-1, M, 2)
    x = F.softmax(x, dim=2)
    x = x + 1e-8
    return x

  def learn(self, x):
    x_pred = self.dec(self.enc(x))
    cost = xentropy_cost(x, x_pred)
    print (cost)

    self.optimizer.zero_grad()
    cost.backward()
    self.optimizer.step()



if __name__ == '__main__':
  anet = AutoNet().cuda()
  for _ in range(1000):
    x = gen_data_batch()
    anet.learn(x)

  embedded = anet.enc(all_batch())
  print (embedded.size())
  embedded_np = embedded.data.cpu().numpy()
  print (embedded_np)

  import pickle
  pickle.dump(embedded_np, open( "embedded_np.p", "wb" ) )

