from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình
model = SimpleCNN()
model.load_state_dict(torch.load('fashion_model_fixed.pth', weights_only=True))
model.eval()

# Định nghĩa transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Khai báo danh sách labels
labels = [
    'Áo thun', 'Quần áo', 'Giày', 'Túi xách', 
    'Quần dài', 'Túi xách', 'Đồ lót', 'Đồ bơi', 
    'Áo khoác', 'Giày thể thao'
]

# Hàm dự đoán
def predict(image):
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    image = Image.open(file.stream).convert('L')  # Chuyển đổi thành ảnh xám
    prediction = predict(image)
    
    # Kiểm tra và trả về dự đoán
    if 0 <= prediction < len(labels):
        return f'Dự đoán: {labels[prediction]}'
    else:
        return f'Dự đoán không hợp lệ: {prediction}'

if __name__ == '__main__':
    app.run(debug=True)
