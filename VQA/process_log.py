import re
from typing import Optional, List, Union
import numpy as np
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


def plot_multiple_lines(
    lines: List[Union[List[float], np.ndarray]],
    labels: List[str],
    x_start: int | None,
    pad_prefix: bool,
    x_step: int,
    xlabel: str,
    ylabel: str,
    title: str,
    need_grid: bool = False,
    save_path: str | None = None
) -> None:
    """
    Plot multiple lines on a graph and optionally save it as an image.

    Args:
        lines: A list containing the data for each line to be plotted. The data can be a list or NumPy array.
        labels: A list containing the labels for each line, used for the legend.
        x_start (optional): The start index of the x-axis. Index starts at 1, not 0, so x_start must be >= 1.
        pad_prefix: When x_start > 1, whether it needs to be padded in front of the sequence.
        x_step: The step of range on x-axis.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        title: The title of the graph.
        need_grad: Whether to make grid in picture.
        save_path (optional): The path to save the image .
    """

    # clip the lines
    if x_start is None:
        x_start = 1
    elif x_start < 1:
        raise ValueError('x_start must be >= 1')
    else:
        for idx, line in enumerate(lines):
            lines[idx] = line[x_start - 1:]
    
    # pad prefix
    if pad_prefix and x_start > 1:
        for idx, line in enumerate(lines):
            prefix = [None] * (x_start - 1)
            prefix.extend(line)
            lines[idx] = prefix
    
    # get series of color
    colors = plt.cm.tab10.colors

    # Find the maximum length among the lines
    max_length = max(len(line) for line in lines)

    # plot x range
    plt.xticks([i for i in range(x_start, max_length, x_step)])

    # Plot the lines
    for idx, (line, label) in enumerate(zip(lines, labels)):
        if len(line) < max_length:
                # Fill missing values with None
                line = np.append(line, [None] * (max_length - len(line)))
                plt.plot(line, label = label, linestyle = 'solid', color = colors[idx])
        else:
            plt.plot(line, label = label, linestyle = 'solid', color = colors[idx])

    # make grid
    if need_grid:
        plt.grid()

    # Add legend and labels
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Save the image (if save_path is provided)
    if save_path:
        plt.savefig(save_path)

    # Display the graph
    plt.show()

    # Clear the current figure
    plt.clf()
