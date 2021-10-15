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
from emb_patches import PatchEmbedding
from attention import MultiHeadAttention
from transformerencoderblock import TransformerEncoderBlock


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
patches_embedded = PatchEmbedding()(x)
print("patch embedding shape is {}".format(patches_embedded.shape))
mha = MultiHeadAttention()(patches_embedded)
print("mha shape is {}".format(mha.shape))
teb = TransformerEncoderBlock()(patches_embedded)
print("TEB shape is {}".format(teb.shape))
