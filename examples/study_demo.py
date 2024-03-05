import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

### CHANGE START:
generic = True
if generic:
    data = np.load('/home/chri6578/Documents/gttp/temp/X_6_log.npz')
    # Convert the data to a PyTorch tensor
    true_y = torch.from_numpy(data['arr_0'][:,0:2])
    true_y = true_y.to(device, torch.float32)
    true_y = true_y.unsqueeze(1)
    true_y0 = true_y[0]
    # print(true_y.shape)
    t = torch.linspace(0., 25., args.data_size).to(device)
    # print(t.shape)

else:
    true_y0 = torch.tensor([[2., 0.]]).to(device)
    t = torch.linspace(0., 25., args.data_size).to(device)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    # this function simulates the real data
    class Lambda(nn.Module):
        def forward(self, t, y):
            return torch.mm(y**3, true_A)


    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, t, method='dopri5')
        print(true_y.shape)
# we can simply initialize true_y with the real data and run this code

### CHANGE END:

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # `batch_y0` is selecting a random subset of true solutions `true_y` at specific indices `s` to
    # create a batch of initial values for the neural network model. It is used as the initial
    # condition for solving the ODE for each batch during training. (batch_size:M, num_dimensions:D).
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('$ \mathbf{x} $') # 2D vector.
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-', label='true')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--', label='fit')
        # ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('$ \mathbf{x}_0 $')
        ax_phase.set_ylabel('$ \mathbf{x}_1 $')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label='true')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--', label='fit')
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('$ \mathbf{x}_0 $')
        ax_vecfield.set_ylabel('$ \mathbf{x}_1 $')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # need to change the dimensios here
        # d x 
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Linear(50, 100),
            nn.Linear(100, 2),
        )
        
        # for name, param in self.net.named_parameters():
        #     print(f"Parameter '{name}' has datatype: {param.dtype}")

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)

    def forward(self, t, y): # need to incorporate t
        print(t, y.shape)
        
        # ISSUE: t should have the same length as y.shape[0]
        # if len(y.shape)==3:
        #     t_vec = t.unsqueeze(1).unsqueeze(2)
        #     y_t = torch.cat((y, t_vec), dim=2)
        
        # else:
        #     t_vec = t.unsqueeze(1)
        #     y_t = torch.cat((y, t_vec), dim=1)
        
        
        # print(y.shape) # batch_size, 1 , dim
        # t_tensor = torch.Tensor(t)
        # t_vec = t_tensor.repeat(args.batch_size, 2)
        # print(y.shape, t_vec.shape)
        
        return self.net(y) # why is the ODE defined?


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device) # important line
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
