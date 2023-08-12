import re
from typing import Optional, List
import matplotlib
import matplotlib.pyplot as plt


def get_loss(record: str) -> Optional[float]:
    pattern = r'loss: (\d+\.\d+)'
    match = re.search(pattern, record)
    if match:
        return float(match.group(1))
    else:
        return None


def get_losses(filepath: str) -> List[float]:
    with open(filepath, 'r') as f:
        logs = f.readlines()
    
    res = []
    for record in logs:
        loss = get_loss(record)
        res.append(loss)
    return res


def plot_loss(loss: List[float], plot_filepath: str, color: str, title: str, plot_step: int = 10):
    # setting
    plt.figure(figsize=(25, 7))
    
    # plot
    y = loss
    x = [i for i in range(1, len(y) + 1)]
    plt.plot(x, y, c=color, marker='D', markersize=2)
    
    # loss print
    loss_print_idx_list = [i for i in range(0, len(x), plot_step)]
    for idx in loss_print_idx_list:
        plt.text(x[idx], y[idx], f"{y[idx]:.2f}", ha='center', va='bottom', fontsize=10)
    # if (len(x) - 1) % plot_step != 1:
    #     last_idx = len(x) - 1
    #     plt.text(x[last_idx], y[last_idx], f"{y[last_idx]:.2f}", ha='center', va='bottom', fontsize=10)
    
    # axis
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    
    # save image
    plt.savefig(plot_filepath)

plot_loss(
    get_losses(r'/gly/guogb/lym/train.log'),
    '/gly/guogb/lym/train-loss.svg',
    'b',
    'Loss of Training'
)

plot_loss(
    get_losses(r'/gly/guogb/lym/val.log'),
    '/gly/guogb/lym/val-loss.svg',
    'r',
    'Loss of Valuation'
)
