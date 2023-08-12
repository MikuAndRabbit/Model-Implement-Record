import os
import torch
import torchvision
from typing import List
from tqdm import tqdm
from data import DVAE_Dataset
from modeling_discrete_vae import DiscreteVAE

def reconstruction(image_paths: List[str], save_path: str, model):
    dataset = DVAE_Dataset(image_paths, 224)
    for idx in tqdm(range(len(dataset))):
        input_image = dataset[idx]
        filename = image_paths[idx].split('/')[-1].split('.')[0]
        torchvision.utils.save_image(input_image, os.path.join(save_path, filename + '_input.jpg'))
        input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            reconstruction_image = model.forward(input_image, return_loss=False)
        torchvision.utils.save_image(reconstruction_image[0], os.path.join(save_path, filename + '_reconstruction.jpg'))


if __name__ == "__main__":
    checkpoint_path = r''
    image_resource_path = r''
    target_path = r''
    
    model_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    model = DiscreteVAE(image_size=224, num_layers=4, num_tokens=8192, codebook_dim=1024, hidden_dim=512)
    model.load_state_dict(model_dict)

    image_paths = []
    for root, dirs, files in os.walk(image_resource_path):
        for file in files:
            image_paths.append(os.path.join(root, file))
        break

    reconstruction(image_paths, target_path, model)
