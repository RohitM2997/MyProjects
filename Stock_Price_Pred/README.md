## Stock Price Prediction

- **Dataset**
    
  - **Dataset Name:** Stock price prediction
  - **Source:** Drive CSV file


- **Data Preprocessing:**
  - Observed stock data and slected one stock with high data for better model training
  - Performed data cleaning steps to get only required and importatnt data
  
- **Requirements:**
  - Install the required dependencies using:
  - pip install tensorflow pandas matplotlib seaborn 
  - Key Libraries:
  - TensorFlow/Keras
  - Pandas
  - Matplotlib
  - Seaborn


# LSTM Model

- **Architecture:**
  - One LSTM layer.
  - One output layer.
  - Dropout applied to prevent overfitting.

- **Compilation:**
  
  - **Optimizer:** Adam 
  - **Loss Function:** Mean Squared Error.
    

### Results

| LSTM Model    | Training data |  Test data    |
|---------------|---------------|---------------|
|     RMSE      |      61.83    |               | 
|---------------|---------------|---------------|
|     MAE       |      49.82    |               | 
|---------------|---------------|---------------|
|    R2 Score   |      0.210    |    0.090      | 
|---------------|---------------|---------------|


 
