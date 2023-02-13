from pathlib import Path
import cv2

from rl.expert.utils import TransitionsDataset
from rl.models.vit import ViT

from torch.utils.data import DataLoader
from torchvision import transforms


from cathsim.utils import make_env

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: ((x / 255.) - 0.5) * 2),
])

trial_path = Path.cwd() / 'rl' / 'expert' / 'trial_2'
transitions_ds = TransitionsDataset(trial_path, transform=transforms)
transitions_dl = DataLoader(transitions_ds, batch_size=32, shuffle=True)

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

v.load_from_checkpoint(file_path / 'trained' / 'vit.ckpt')

wrapper_kwargs = dict(
    time_limit=300,
    use_pixels=True,
    use_obs=[
        'pixels',
    ],
    grayscale=True,
    resize_shape=128,
    frame_stack=4,
)

env = make_env(wrapper_kwargs=wrapper_kwargs)
obs = env.reset()
done = False
while not done:
    act = v(obs)
    obs, rew, done, info = env.step(act)
    image = env.render(mode='rgb_array')
    cv2.imshow('image', image)
    cv2.waitKey(1)
