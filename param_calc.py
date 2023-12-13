from Student_Models import unet, unet_attention, unet_inception
from Models import models
import torch.nn as nn

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
    model (nn.Module): A PyTorch model.

    Returns:
    int: Total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate models
unet128 = unet.UNet(3, 1, [64, 128])
unet256 = unet.UNet(3, 1, [64, 128, 256])
unet512 = unet.UNet(3, 1, [64, 128, 256, 512])

unet_attention128 = unet_attention.UNetAttn(3, 1, [64, 128])
unet_attention256 = unet_attention.UNetAttn(3, 1, [64, 128, 256])
unet_attention512 = unet_attention.UNetAttn(3, 1, [64, 128, 256, 512])

unet_inception128 = unet_inception.UNetInception(3, 1, [64, 128])
unet_inception256 = unet_inception.UNetInception(3, 1, [64, 128, 256])
unet_inception512 = unet_inception.UNetInception(3, 1, [64, 128, 256, 512])

transformer_model = models.FCBFormer()

# Prepare the models dictionary
models_dict = {
    'UNet 128': unet128,
    'UNet 256': unet256,
    'UNet 512': unet512,
    'UNet Attention 128': unet_attention128,
    'UNet Attention 256': unet_attention256,
    'UNet Attention 512': unet_attention512,
    'UNet Inception 128': unet_inception128,
    'UNet Inception 256': unet_inception256,
    'UNet Inception 512': unet_inception512,
    'Transformer Model': transformer_model
}

# Calculate parameters for each model
params_count = {name: count_parameters(model) for name, model in models_dict.items()}

# Find the maximum parameter count
max_params = max(params_count.values())

# Normalize and write parameters of each model to a file
with open('model_parameters_normalized.txt', 'w') as file:
    for name, count in params_count.items():
        normalized_count = count / max_params
        file.write(f"{name}: {normalized_count:.4f} (normalized)\n")

print("Normalized model parameters have been saved to 'model_parameters_normalized.txt'")
