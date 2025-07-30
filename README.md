# Handwritten_Numbers_Recognition
uses pytorch torchvision opencv-python matplotlib numpy to predict the number feeded into the model 
Handwritten Digit Recognition(PyTorch)

This project implements a digit classifier using the MNIST dataset in PyTorch, and also supports prediction on **custom digit images**(like i have done)

Requirements 
Install all required libraries via pip:

bash
pip install torch torchvision opencv-python matplotlib numpy

Project Structure
.
├── digit_recognizer.py         # Main Python script
├── README.md                   # This file
└── digits/                     # Your custom test images
    ├── digit1.png
    ├── digit2.png
    ├── ...

Dataset

Training dataset**: MNIST is automatically downloaded using `torchvision.datasets`.
**Custom test images**: Place your PNG images in the `digits/` folder.
  - Format: `digitX.png` (e.g. `digit1.png`, `digit2.png`, ...)
  - Size: 28x28 pixels (or any size; the script resizes automatically)
  - Format: Black digit on white background (or inverted; handled in code)

How to Run

1. Clone or Download the Repo
`bash
git clone https://github.com/yourname/digit-classifier.git
cd digit-classifier

2. Run the Classifier
`bash
python digit_recognizer.py

What This Does

1. Load and Normalize the Dataset
`python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
`
- MNIST is automatically downloaded and normalized to [-1, 1].

2. Define the Neural Network
`python
self.fc1 = nn.Linear(28*28, 128)
self.fc2 = nn.Linear(128, 128)
self.fc3 = nn.Linear(128, 10)
`
- A fully connected neural network with 2 hidden layers.

3. Train the Model
`python
model.fit(..., epochs=3)
`
- Trains on MNIST digits (60,000 images).
- Saves the model as `digit_classifier.pth`.

4. Evaluate Accuracy
`python
Test Accuracy: 93.89%

5. Predict Your Custom Digits
`python
digits/digit1.png
digits/digit2.png
.
`
- Resizes to 28x28, normalizes, inverts
- Predicts using trained model
- Shows image with predicted label

Example Output
Training the model...
Epoch 1 complete. Loss: 0.23
...
Test Accuracy: 93.91%
Image 1: Probably a 4
Image 2: Probably a 9
...

License

MIT License. Feel free to fork and build upon!
Created By Aditya Prasad Panigrahi
