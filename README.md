🍕 Pizza vs. Not Pizza: Deep Learning Image Classifier
This project demonstrates an end-to-end Computer Vision pipeline using TensorFlow and Keras to classify images. The project culminates in a real-time web application built with Gradio that allows users to test the model with their own images.

📌 Project Overview
The goal of this project is to build a binary classifier capable of distinguishing between images of pizzas and other objects. This involves handling a real-world dataset, implementing a Convolutional Neural Network (CNN), and deploying the model for interactive use.

🛠️ Tech Stack
Language: Python

Deep Learning: TensorFlow / Keras

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Deployment: Gradio

📊 Dataset Details
Total Images: 1,966

Classes: * pizza: 983 images

not_pizza: 983 images

Preprocessing: Images were resized to 224x224 pixels and normalized to a [0, 1] range using ImageDataGenerator.

🏗️ Model Architecture
The project utilizes a custom Convolutional Neural Network (CNN) designed for binary classification:

Multiple Conv2D layers for feature extraction.

MaxPooling2D layers for downsampling.

Dropout layers to prevent overfitting and improve generalization.

A Dense output layer with a Sigmoid activation function.

🚀 How to Run Locally
1. Download the Data
Because the dataset is large, it is hosted in the Releases section of this repository.

Download pizza.zip from the latest release.

Place it in the same directory as the notebook.

2. Install Requirements
Bash

pip install tensorflow gradio numpy pandas matplotlib seaborn
3. Execution
Open Image_project.ipynb in Google Colab or Jupyter Notebook and run all cells. The notebook is configured to:

Extract the dataset.

Train the CNN model.

Launch a Gradio web interface.

🌐 Deployment
The final model is wrapped in a Gradio interface, providing a user-friendly way to:

Upload any image from your computer.

See the model's confidence score for both "Pizza" and "Not Pizza" categories.

Developed as a portfolio project to showcase expertise in Deep Learning and Model Deployment.
