import torch
import torchvision.models as models
from unet import UNet  # Assume your U-Net implementation here

def load_classification_model():
    model = models.densenet121(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 9)
    model.load_state_dict(torch.load("model/densenet_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def load_segmentation_model():
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("model/unet_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model
