from typing import List
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as F


def preprocess_image(image: Image.Image, image_size: int) -> Image.Image:
    # 将图片随机裁剪成正方形
    w, h = image.size
    small_size = min(h, w)
    img = transforms.RandomCrop(small_size)(image)
    # 将图片缩放到一个随机大小
    t_min = min(small_size, round(9 / 8 * image_size))
    t_max = min(small_size, round(12 / 8 * image_size))
    t = torch.randint(low=int(t_min), high=int(t_max + 1), size=(1,))
    img = F.resize(img, [int(t.item()), int(t.item())])
    # 再将图片随机裁剪成指定大小并进行填充
    img = transforms.RandomCrop(image_size)(img)
    img = F.hflip(img)
    return img  # type: ignore

class DVAE_Dataset(Dataset):
    def __init__(self, img_urls: List, image_size: int, augumentation: bool = False) -> None:
        super().__init__()
        self.img_urls = img_urls
        self.image_size = image_size
        
        self.augumentation = augumentation
        if self.augumentation:
            self.image_transformer = transforms.ToTensor()
        else:
            self.image_transformer = transforms.Compose([
                transforms.Resize([self.image_size, self.image_size]),
                transforms.ToTensor()
            ])
        
    def __len__(self) -> int:
        return len(self.img_urls)
    
    def __getitem__(self, index):
        img_url = self.img_urls[index]
        image = Image.open(img_url)
        # check mode of image
        if image.mode != 'RGB':
            image = image.convert(mode = 'RGB')
        # augumentation
        if self.augumentation:
            image = preprocess_image(image, self.image_size)
        return self.image_transformer(image)

def judge_image_mode(image_paths: List[str]):
    res = {}
    from tqdm import tqdm
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        mode = image.mode
        if res.get(mode) is None:
            res[mode] = 1
        else:
            res[mode] += 1
        image.close()
    return res

