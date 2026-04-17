import torch
from pathlib import Path
import matplotlib.pyplot as plt

file_dir= Path("testing/subtomos/val_subtomos/subtomo0/0.pt")

tensor = torch.load(file_dir, weights_only= True)

image = tensor.numpy()

plt.imshow(image)
plt.show()