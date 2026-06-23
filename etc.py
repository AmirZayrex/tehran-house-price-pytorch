import torch
state = torch.load("Artifacts/pytorch_model.pth", map_location="cpu")
for key, value in state.items():
    print(f"{key}: {value.shape}")