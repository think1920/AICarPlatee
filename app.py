import os
from flask import Flask, render_template, request, redirect, url_for
from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

# Kết nối Roboflow
rf = Roboflow(api_key="xxxx")
project = rf.workspace().project("xxx")
model = project.version(2).model

# Thư mục lưu ảnh tải lên và ảnh cắt
UPLOAD_FOLDER = 'static/uploads/'
CROPPED_FOLDER = 'static/cropped/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Lưu ảnh tải lên
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Dự đoán từ mô hình Roboflow
        result = model.predict(filepath, confidence=40, overlap=30).json()

        # Mở ảnh gốc
        image = Image.open(filepath)

        # Cắt ảnh biển số và trích xuất văn bản
        cropped_image_file = None
        text = None

        for i, prediction in enumerate(result['predictions']):
            x_center = prediction['x']
            y_center = prediction['y']
            width = prediction['width']
            height = prediction['height']

            left = x_center - width / 2
            top = y_center - height / 2
            right = x_center + width / 2
            bottom = y_center + height / 2

            cropped_image = image.crop((left, top, right, bottom))
           
            resize_test_license_plate = cv2.resize(np.array(cropped_image), None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
            grayscale_resize_test_license_plate = cv2.cvtColor(resize_test_license_plate, cv2.COLOR_BGR2GRAY) 
            gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0) 
            
            # Lưu ảnh cắt vào thư mục 'cropped'
            cropped_filename = f"cropped_{i}.png"
            cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
            cropped_image.save(cropped_path)
            cropped_image_file = cropped_filename

            # Trích xuất văn bản từ ảnh biển số
            lang = 'eng'
            config = r'--psm 8 --oem 3 -c tessedit_char_whitelist= ABCDEFGIJKLMNOPQRSTUVWXYZ0123456789-/'
            text = pytesseract.image_to_string(gaussian_blur_license_plate, lang=lang, config=config)

        # Trả về template với ảnh đã cắt và văn bản
        return render_template('index.html', image=cropped_image_file, text=text)

if __name__ == '__main__':
    app.run(debug=True)
