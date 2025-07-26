
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

def load_resnet50_v2(device=None):
    """
    Loads ResNet-50 with the IMAGENET1K_V2 pretrained weights (alias DEFAULT).
    """
    weights = ResNet50_Weights.DEFAULT  # same as IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    if device:
        model = model.to(device)
    return model, weights

def preprocess_image(img_path, weights):
    """
    Preprocess an image for ResNet-50 (V2), using transforms bundled with the weights.
    """
    img = Image.open(img_path).convert("RGB")
    preprocess = weights.transforms()
    tensor = preprocess(img).unsqueeze(0)  # add batch dim
    return tensor

def predict(img_path, model, weights, device=None):
    """
    Runs inference and returns top-1 prediction label and confidence.
    """
    input_tensor = preprocess_image(img_path, weights)
    if device:
        input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        class_idx = probs.argmax().item()
        score = probs[class_idx].item()
        label = weights.meta["categories"][class_idx]
    return label, score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weights = load_resnet50_v2(device)

    img_path = "path/to/your/image.jpg"
    label, confidence = predict(img_path, model, weights, device)
    print(f"Prediction: {label} ({confidence * 100:.1f}%)")