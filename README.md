# neural-recognition-neighbour
neural-recognition-neighbour

About

neural-recognition-neighbour is a neural network designed for recognizing neighbors' faces from photographs. The network employs PyTorch for creating, training, and testing the model.
Features

    Built upon a Deep Convolutional Neural Network (CNN).
    Data comprises .jpg format images.
    Recognizes faces of four neighbors: Kirill, Oleg, Vlad, and Roma.
    Has capabilities for further data augmentation for improved training.
    The trained model is saved and can be loaded for further use or continued training.

Installation and Usage

    Clone the repository from GitHub.
    Ensure you have all the necessary libraries installed (PyTorch, torchvision, numpy, pandas, among others).
    Place your images in the archive/seg_train folder for training and archive/seg_pred for testing.
    Run the script. If you want to train the model from scratch, set train = True.

Project Structure

    PyTorch is utilized for model creation and training.
    Images are loaded using torchvision and PIL.
    Data preprocessing includes image resizing, augmentation, and tensor conversion.
    The model comprises three convolutional layers followed by fully connected layers.
    Adam optimizer is employed for optimization, and the loss function used is CrossEntropyLoss.
    Post training, the model's effectiveness can be tested on a test data set.

Contributing

Any contributions to enhance or develop the project further are welcome! Create a pull request or discuss ideas in the issues section.
License

This project is distributed under the MIT License. Details can be found in the LICENSE file.

This is a basic project description based on the provided code. You can add more details if deemed necessary.
