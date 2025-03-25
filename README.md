# Image Editor Application

This repository contains an image editing application built with Python. It uses Tkinter for the GUI and libraries such as Pillow, Matplotlib, and NumPy for image processing. The app allows users to open an image, apply various filters (like grayscale, negative, binarization, blur, and edge detection), adjust brightness/contrast, and view image histograms and projections. This is a first project for Biometrics course at Faculty of Mathematics and Information Science at Warsaw University of Technology.

## Features

### Image Operations
- Open and save images
- Convert to grayscale
- Apply negative and binarization effects

### Filters
- Average Blur, Gaussian Blur, and Sharpen
- Edge detection using Roberts, Sobel, and Prewitt kernels
- Custom filter kernel input

### Adjustments
- Brightness and contrast sliders for fine-tuning

### Visualization
- Display histograms and vertical/horizontal projections for image analysis

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AristocratesJ/Simple-Image-Editor.git
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
cd src
python main.py
```

## Usage

When you launch the application, it automatically loads a sample image located at `test/imgs/dog.jpg`. Use the menu bar to:
- Open a new image or save your edited image.
- Apply operations like grayscale, negative, or binarize.
- Experiment with various filters and adjust image parameters using the sliders.
- Generate plots such as histograms and projections to analyze the image.

## Documentation

Documentation is available [there](./Dokumentacja_aplikacji_projekt_1_biometria.pdf)
