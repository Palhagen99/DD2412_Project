import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from data_loader import *
from models_mae import *
from train import *


data_path = './UCF-101/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 6

transforms = v2.Compose([
    v2.Resize(size=(224,224), antialias=True), # Resize for ViT
    v2.Lambda(lambd=lambda x: x/255.0) # Normalize
])

train_data = UCF101FullVideo(root=data_path, output_format="TCHW", transform=transforms)
train_loader = DataLoader(train_data, 8, shuffle=True, collate_fn=custom_collate, pin_memory=True, num_workers=6)
model= nn.DataParallel(sim_mae_vit_tiny_patch8_dec512d8b(use_joint_enc=False, use_joint_dec=False)) 
model = model.to(device)
model = train(model, train_loader, num_epochs=num_epochs, lr=1e-4, name="SIAM_CROSS_SELF") 