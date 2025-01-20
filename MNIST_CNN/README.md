## MNIST Digit Recognition

- **Dataset**
    
  - **Dataset Name:** MNIST Digit Recognition
  - **Source:** Drive CSV file


- **Data Preprocessing:**
  - **Images are resized to 255x255 pixels.**
  
- **Requirements:**
  - Install the required dependencies using:
  - pip install tensorflow pandas matplotlib seaborn kaggle
  - Key Libraries:
  - TensorFlow/Keras
  - Pandas
  - Matplotlib
  - Seaborn


# Custom CNN

- **Architecture:**
  - Two Convolutional layers with ReLU activation.
  - Batch Normalization and MaxPooling for regularization and dimensionality reduction.
  - Fully connected layers with a final output layer using the softmax activation function.
  - Dropout applied to prevent overfitting.

- **Compilation:**
  
  - **Optimizer:** Adam 
  - **Loss Function:** Catagorical Cross-Entropy.
  - **Metrics:** Accuracy.
    

### Results

| Model         | Test Accuracy |
|---------------|---------------|
| Custom CNN    | ~99%          | 




 
