# Car Damage Detection System 🚗

A smart application that helps detect and classify car damage using artificial intelligence. Simply upload a photo of your car, and the system will tell you if there's any damage and what type it is.

![App Screenshot](streamlit-app/app_screenshot.jpg)

## 📱 What Can This App Do?

- Takes a photo of your car (front or rear view)
- Analyzes the image instantly
- Tells you if there's any damage
- Classifies the type of damage (normal, crushed, or breakage)
- Works through an easy-to-use web interface

## 🎯 Accuracy

The system is about 80% accurate in detecting car damage, trained on 1700+ real car images.

## 📂 Project Structure

The project has three main parts:

1. **training/** 
   - Contains the deep learning model training code
   - Uses ResNet50 for accurate damage detection
   - Jupyter notebook with step-by-step training process

2. **streamlit-app/**
   - User-friendly web interface
   - Simple drag-and-drop image upload
   - Clear damage detection results

3. **fastapi-backend/**
   - Fast and reliable backend server
   - Handles image processing
   - Returns damage predictions

## 🔍 Damage Types It Can Detect

- Front Normal
- Front Crushed
- Front Breakage
- Rear Normal
- Rear Crushed
- Rear Breakage

## 💻 How to Run the App

1. First, install all required packages:
```bash
pip install -r requirements.txt
```

2. Start the backend server:
```bash
cd fastapi-backend
uvicorn server:app --reload
```

3. Launch the web interface:
```bash
cd streamlit-app
streamlit run app.py
```

## 📸 Taking Good Photos

For best results:
- Take photos in good lighting
- Capture the car from a three-quarter angle
- Make sure the damage area is clearly visible
- Avoid shadows or reflections

## 🛠️ Technical Details

- **Model**: ResNet50 (Transfer Learning)
- **Training Data**: 1700+ car images
- **Accuracy**: ~80% on validation set
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Image Processing**: Python with Deep Learning

## 🤝 Need Help?

If you have questions or run into problems:
1. Check if all packages are installed correctly
2. Make sure you're using the right photo angle
3. Verify both backend and frontend are running

## 📋 System Requirements

- Python 3.7 or higher
- Minimum 4GB RAM
- Webcam or ability to upload images
- Internet connection for first-time package installation

---
Made with ❤️ for making car damage detection easier