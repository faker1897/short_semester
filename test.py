import torch, torchvision
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Available:", torch.cuda.is_available())
print("TorchVision:", torchvision.__version__)
# 测试 GPU NMS
boxes  = torch.rand(5, 4, device='cuda')
scores = torch.rand(5,    device='cuda')
print("NMS output:", torchvision.ops.nms(boxes, scores, 0.5))
