from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = torch.jit.load("model_scripted.pt", map_location=torch.device("cpu"))
model.eval()

class_names = ['Bees', 'Beetles', 'Butterfly', 'Cicada', 'Dragonfly',
               'Grasshopper', 'Moth', 'Scorpion', 'Snail', 'Spider']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_single_image(image):
    try:
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = probabilities.argmax().item()
            confidence = probabilities[predicted_class_idx].item()

        return class_names[predicted_class_idx], round(confidence * 100, 2)
    except Exception as e:
        return f"Error: {e}", None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    if file.filename == '':
        return {"error": "No file selected"}, 400

    try:
        image = Image.open(file.stream).convert("RGB")
        width, height = image.size

        predicted_class, confidence = predict_single_image(image)

        bounding_box = [50, 50, width - 50, height - 50]  # [x_min, y_min, x_max, y_max]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "bounding_box": bounding_box
        }
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)
