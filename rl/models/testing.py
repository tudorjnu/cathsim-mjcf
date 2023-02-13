from vit_pytorch.recorder import Recorder
import torch
from vit_pytorch.vit import ViT

from pathlib import Path
import numpy as np
from rl.expert.utils import TransitionsDataset


from torch.utils.data import DataLoader
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: ((x / 255.) - 0.5) * 2),
])

trial_path = Path.cwd() / 'rl' / 'expert' / 'trial_2'
transitions_ds = TransitionsDataset(trial_path, transform=transforms)
transitions_dl = DataLoader(transitions_ds, batch_size=32, shuffle=True, num_workers=16)

file_path = Path(__file__).parent

v = ViT(
    image_size=128,
    patch_size=16,
    num_classes=2,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1028,
    channels=1,
    emb_dropout=0.1
)

# train the model

optimizer = torch.optim.Adam(v.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(1):
    for obs, acts in transitions_dl:
        optimizer.zero_grad()
        preds = v(obs)
        loss = criterion(preds, acts)
        loss.backward()
        optimizer.step()
        print(loss.item())


obs, acts = next(iter(transitions_dl))
print(obs)
v = Recorder(v)
preds, attns = v(obs)
print(attns)

np.savez(file_path / 'trained' / 'attns.npz', obs=obs, attns=attns)
