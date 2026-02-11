import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("dashiki_kurti_model.pth"))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["dashiki", "kurti"]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

if __name__ == "__main__":
    img_path = sys.argv[1]
    print()
    print()
    print("\t\t\tPrediction:", predict(img_path))
