# License Plate Recognition with Flask, Roboflow, and Tesseract OCR

This project is a web application built using Flask that allows users to upload vehicle images, automatically detects the license plates using a Roboflow-trained object detection model, and extracts text from the license plates using Tesseract OCR.

## Features

- Upload vehicle images through a simple web interface.
- Detect license plates in uploaded images using Roboflow's hosted model.
- Crop and preprocess detected license plates.
- Extract license plate numbers using Tesseract OCR.
- Display the cropped plate image and the extracted text.

## Technologies Used

- **Flask** – for building the web application.
- **Roboflow** – for license plate detection.
- **Tesseract OCR** – for optical character recognition (OCR).
- **OpenCV** – for image preprocessing.
- **Pillow (PIL)** – for image manipulation.

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/license-plate-recognition.git
   cd license-plate-recognition
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt (skip it if no file)
   ```

3. **Configure Tesseract**:

   Make sure Tesseract OCR is installed and available in your system's PATH.  
   Download: https://github.com/tesseract-ocr/tesseract

4. **Set your Roboflow API key**:

   In `app.py`, replace:

   ```python
   rf = Roboflow(api_key="YOUR_API_KEY")
   ```

   with your actual API key from https://roboflow.com/.

5. **Run the application**:

   ```bash
   python app.py
   ```

6. **Visit the web interface**:

   Open your browser and go to:  
   http://127.0.0.1:5000

## Example Workflow

1. User uploads a vehicle image.
2. App uses Roboflow to detect the license plate.
3. Detected plate is cropped and processed with OpenCV.
4. Tesseract OCR extracts the text from the plate.
5. The result (plate image + extracted text) is shown on the webpage.
