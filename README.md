# X-Ray Image Inpainting

## Project Overview
This project focuses on classifying chest X-ray images into two categories: Normal and Pneumonia. It leverages a pre-trained VGG19 convolutional neural network model with custom layers added for classification. The model is trained using TensorFlow and Keras, with data augmentation applied to improve generalization. Additionally, Grad-CAM visualization is used to interpret the model's predictions.

## Dataset Structure
The dataset is expected to be organized in the following directory structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Each folder contains the respective chest X-ray images for training, validation, and testing.

## Installation

1. Clone the repository or download the project files.
2. It is recommended to create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Open the Jupyter notebook `X_RAY_IMAGE_IN_PAINTING.ipynb`.
- Ensure the dataset is placed in the correct directory structure as described above.
- Run the notebook cells sequentially to train the model, evaluate performance, and visualize results using Grad-CAM.

## Model Details

- Base model: VGG19 pre-trained on ImageNet (with frozen layers).
- Custom layers: Flatten, Dense (512 units, ReLU), Dropout (0.5), and final Dense layer with softmax activation for 2 classes.
- Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.0001 and momentum 0.9.
- Loss function: Categorical Crossentropy.
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.

## Evaluation

- Model accuracy and loss are plotted for training and validation sets.
- Confusion matrix and classification report are generated for test data.
- Grad-CAM visualizations highlight important regions in the X-ray images influencing the model's decisions.

## License

This project is provided as-is for educational and research purposes.
