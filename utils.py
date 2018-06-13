import torch
from torch.autograd import Variable
def to_torch(x, req = False):
  x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad = req)
  return x

# simple cross entropy cost (might be numerically unstable if pred has 0)
def xentropy_cost(x_target, x_pred):
  assert x_target.size() == x_pred.size(), \
      "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
  logged_x_pred = torch.log(x_pred)
  cost_value = -torch.sum(x_target * logged_x_pred)
  return cost_value

