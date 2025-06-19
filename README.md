# Furniture Image Classifier

A web application for classifying images of furniture into four categories: **Chair**, **Sofa**, **TV**, and **Table**. The app provides a user-friendly interface to upload images and select between two deep learning models (CNN and ResNet) for prediction.

## Features

- Upload an image of furniture and get an instant prediction.
- Choose between a custom CNN model and a ResNet-based model.
- Clean, modern web interface built with Flask and HTML/CSS.
- Pre-trained models included for immediate use.

## Getting Started

### Prerequisites

- Python 3.1 | https://www.python.org/downloads/release/python-3116/
- pip

### Installation

1. **Clone the repository:**
   git clone <repo-url>
   cd Furniture_Image_Classifier

2. **Install dependencies:**
   pip install flask pillow tensorflow


3. **Ensure models are present:**
   - The `models/` directory should contain `furniture_model_CNN.h5` and `furniture_model_RESNET.h5`.
   - If not, train your own using the notebooks in `notebooks/`.

### Running the App

python app.py
- The app will be available at `http://localhost:9000/`.

## Usage

1. Open the web app in your browser.
2. Select the model (CNN or ResNet).
3. Upload an image of a piece of furniture.
4. Click "Classify Image".
5. View the predicted class and the model used.

## Datasets

- The `datasets/` folder contains images organized by class:
  - `Chair_Images/`
  - `Sofa_Images/`
  - `TV_Images/`
  - `Table_Images/`

## Model Training
- Use the Jupyter notebooks in the `notebooks/` directory to train or fine-tune models.
- Save trained models as `.h5` files in the `models/` directory.

## Web Interface
- The interface is styled for clarity and ease of use.
- Users can preview the uploaded image and see the prediction result instantly.
