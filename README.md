# Project Documentation

## Project Aim

The aim of this project is to train a model to be able to recognize handwritten letters. This is achieved by using personal handwriting and data that is accessible on the HTL - Anichstraße AI - Server, to train the model.

Furthermore, the development of a GUI is necessary to enable ease of use.

## Structure of the Project

### BigDataSet

    Contains subfolders A - Z with various handwritten letters. This is used to train the model and was accessed from the AI - Server.

### Extrahieren_Buchstaben.py
    A Python script that extracts letters from a formatted handwritten sheet.

### prepare_training_data.py
    Prepares the training data and labels so they can be processed by Tensorflow.

### train_model.py
    Trains the model based on the provided data from prepare_training_data.py

### letter_recognition_model.keras
    The resulting model with an accuracy of 96%.

### GUI.py
    Main program that runs the model and provides the user with a GUI consisting of a canvas. In this canvas, the user can draw a letter, which is then interpreted. The model then guesses the letter and shows the confidence rate.

## Result

![GUI - Screenshot](/assets/GUI.png)