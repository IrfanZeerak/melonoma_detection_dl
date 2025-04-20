import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from .models import load_classification_model

model = load_classification_model()
model.eval()

def generate_gradcam(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    # Hook the gradients and activations
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    target_layer = model.features[-1]  # depends on architecture
    handle = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred = output.argmax(dim=1)
    class_score = output[0, pred]
    model.zero_grad()
    class_score.backward()

    grads = gradients[0].squeeze().detach().numpy()
    acts = activations[0].squeeze().detach().numpy()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
    superimposed = heatmap * 0.4 + img
    handle.remove()

    return Image.fromarray(cv2.cvtColor(np.uint8(superimposed), cv2.COLOR_BGR2RGB))
