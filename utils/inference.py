import torch
import torchvision.transforms as transforms
from PIL import Image
from .models import load_classification_model, load_segmentation_model

# Load models once
clf_model = load_classification_model()
seg_model = load_segmentation_model()

classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions',
           'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_image(image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = clf_model(image)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred = torch.max(prob, 0)
        return classes[pred.item()], confidence.item()

def segment_image(image: Image.Image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = seg_model(img_tensor)
        mask = (output > 0.5).float()
        mask_img = mask.squeeze().numpy()
    return Image.fromarray((mask_img * 255).astype(np.uint8))
