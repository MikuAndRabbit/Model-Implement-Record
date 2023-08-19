import torch
import numpy as np
from tqdm import tqdm
import time
import json
import torch.nn as nn
import torchvision
from PIL import Image


x = torch.zeros(3192)
x[1] = 1

print(sum(x))