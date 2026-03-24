import torch
import argparse
from PIL import Image
from torchvision import models, transforms


def resize_aspect_with_padding(img, target=224):
    w, h = img.size
    scale = float(target) / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    x = (target - nw) // 2
    y = (target - nh) // 2
    canvas.paste(img, (x, y))

    return canvas


def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_image(model, image_path):
    transform = build_transform()
    img = Image.open(image_path).convert("RGB")
    img = resize_aspect_with_padding(img)
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    classes = ["UXO", "non_UXO"]
    pred = torch.argmax(probs).item()
    return classes[pred], float(probs[pred]), probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    label, confidence, probs = predict_image(model, args.image)
    print("Prediction:", label)
    print("Confidence:", confidence)
    print("Probabilities [UXO, non_UXO]:", probs)


if __name__ == "__main__":
    main()
