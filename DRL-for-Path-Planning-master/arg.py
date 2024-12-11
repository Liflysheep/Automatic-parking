import argparse
import torch as th


# 其他必要的导入

def parse_args():
    parser = argparse.ArgumentParser(description="Soft Actor-Critic Hyperparameters")

    # Add agent param
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor γ')
    parser.add_argument('--alpha', type=float, default=0.2, help='Temperature coefficient α')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--update_after', type=int, default=1000, help='Start training after this many steps')

    parser.add_argument('--lr_decay_period', type=int, default=None, help='Learning rate decay period')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for the critic')
    parser.add_argument('--lr_actor', type=float, default=1e-3, help='Learning rate for the actor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target Q')

    parser.add_argument('--q_loss_cls', type=str, default='nn.MSELoss', help='Q loss function type')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping range')

    parser.add_argument('--adaptive_alpha', type=bool, default=True, help='Adaptive alpha flag')
    parser.add_argument('--target_entropy', type=float, default=None, help='Target entropy for adaptive alpha')
    parser.add_argument('--lr_alpha', type=float, default=1e-3, help='Learning rate for alpha')
    parser.add_argument('--alpha_optim_cls', type=str, default='th.optim.Adam', help='Alpha optimizer type')

    parser.add_argument('--device', type=str, default="cuda" if th.cuda.is_available() else "cpu",
                        help='Device for computation')



    return parser.parse_args()


