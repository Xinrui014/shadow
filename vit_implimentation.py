import torch
from model.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)
for name, parameters in v.named_parameters():
    print(name, parameters.size())

preds = v(img) # (1, 1000)
print(preds)