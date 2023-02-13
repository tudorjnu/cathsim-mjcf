import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path

data = np.load(Path(__file__).parent / 'trained' / 'attns.npz', allow_pickle=True)
data = dict(data)
obs = data['obs']
attns = data['attns']
attns  # (32, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)

print(obs[0].shape)

attns = attns[1, :, :, :, :]


def plot_attention_map(att_mat):
    att_mat = np.stack(att_mat, axis=0).squeeze()
    att_mat = np.mean(att_mat, axis=0)

    print(att_mat.shape)


plot_attention_map(attns)

# # make a plt figure with three figures
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].set_title('Input')
# axs[1].set_title('Attention')
# axs[2].set_title('Overlay')
#
# # plot the input image
# axs[0].imshow(obs[0].squeeze(), cmap='gray')
# axs[0].axis('off')
# axs[0].set_aspect('equal')
# axs[1].imshow(attns[0])
# axs[1].axis('off')
#
# plt.imshow(obs[0].squeeze(), cmap='gray')
# plt.show()
