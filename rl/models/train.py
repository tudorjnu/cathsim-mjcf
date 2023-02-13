from pathlib import Path
import numpy as np
from rl.expert.utils import TransitionsDataset
from rl.models.vit import ViT

from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from vit_pytorch.recorder import Recorder


transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x / 255.)),
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
)

logger = TensorBoardLogger(file_path / 'logs', name="vit")
trainer = pl.Trainer(default_root_dir=Path(__file__).parent, accelerator='gpu',
                     devices=1, max_epochs=500, logger=logger)
trainer.fit(model=v, train_dataloaders=transitions_dl)
trainer.save_checkpoint(file_path / 'trained' / 'vit.ckpt')
