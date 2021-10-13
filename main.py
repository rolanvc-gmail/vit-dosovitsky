import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from patches import PatchEmbedding
img = Image.open('./cat.jpg')
fig = plt.figure()
plt.imshow(img)
plt.show()
#resize to imagenet size
transform = Compose([Resize((224,224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0)
print("shape is {}".format(x.shape))
# patch_size = 16
# patches = rearrange(x, 'b c (h1 s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

print("patch shape is {}".format(PatchEmbedding()(x).shape))
