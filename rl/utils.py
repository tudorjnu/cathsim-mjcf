from cathsim.utils import make_env
import torch
from torchvision import transforms
import cv2
from pathlib import Path

from rl.models.vit import ViT


file_path = Path(__file__).parent / 'models'

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
v.eval()


transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: ((x / 255.) - 0.5) * 2),
])


wrapper_kwargs = dict(
    time_limit=300,
    use_pixels=True,
    use_obs=[
        'pixels',
    ],
    grayscale=True,
    resize_shape=128,
)

env = make_env(wrapper_kwargs=wrapper_kwargs)
obs = env.reset()
print(obs.shape)
done = False
with torch.no_grad():
    while True:
        obs = transforms(obs)
        obs = torch.unsqueeze(obs, 0)
        act = v(obs)
        print(act)
        obs, rew, done, info = env.step(act.detach().numpy())
        image = env.render(mode='rgb_array')
        cv2.imshow('image', image)
        cv2.waitKey(1)
