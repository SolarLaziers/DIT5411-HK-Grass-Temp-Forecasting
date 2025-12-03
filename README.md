# DIT5411: RNN and LSTM for Hong Kong Daily Minimum Grass Temperature Forecasting

## Project Overview
This repository showcases the implementation of the DIT5411 Machine Learning project, which investigates and compares the performance of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models in forecasting Hong Kong's daily minimum grass temperature. Using historical data from the Hong Kong Observatory (1980–2024), the models are trained and evaluated on test data spanning January 1, 2025, to October 30, 2025.

### Key Highlights:
- **Comprehensive Data Preprocessing**: Includes handling missing values, scaling, and generating sequences with a 30-day sliding window.
- **Model Architectures**:
    - **Baseline RNN**: A single-layer architecture with 50 units.
    - **Stacked LSTM**: A two-layer architecture with 50 units per layer, incorporating dropout for improved regularization.
- **Performance Metrics**: Evaluation using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE), complemented by visual comparisons of predictions versus actual values.
- **Findings**: The LSTM model outperforms the RNN model, effectively capturing long-term dependencies and seasonal variations in the data.

## Dataset
- **Source**: Daily minimum grass temperature data provided by the Hong Kong Observatory.
- **Processed Data**: Available in the `/data/` directory for reproducibility.

## Setup and Installation
To set up the environment and install the required dependencies, execute the following:
```bash
# Install dependencies
pip install -r requirements.txt
```
Dependencies include TensorFlow, Keras, Pandas, Matplotlib, and Scikit-learn.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DIT5411-HK-Grass-Temp-Forecasting.git
   ```
2. Navigate to the `/notebooks/` directory and execute the notebooks in the following order:
   - `data_preprocessing.ipynb`: Prepares and processes the dataset.
   - `rnn_model.ipynb`: Trains and saves the RNN model.
   - `lstm_model.ipynb`: Trains and saves the LSTM model.
   - `evaluation.ipynb`: Computes evaluation metrics and generates visualizations.

3. View the results:
   - Plots and figures: Located in the `/figures/` directory.
   - Final report: Available in the `/report/` directory.

## Results Summary
- **RNN**: MAE = 2.1°C, RMSE = 2.8°C
- **LSTM**: MAE = 1.5°C, RMSE = 2.0°C

The LSTM model demonstrates superior performance, particularly in capturing long-term patterns during Hong Kong's winter lows and cold snaps.

## Team Members
- **LUK KA CHUN** (ID: 220268421)
- **Liew Wang Yui** (ID: YYYY)
- **Tang Chi To** (ID: 220304866)

## References
- GeeksforGeeks: Predicting Weather Using LSTM
- Related Projects: [climate-temperature-forecasting-with-LSTM](https://github.com/example1), [Time-Series-Forecasting-using-LSTM](https://github.com/example2)

For detailed code and implementation, refer to the Jupyter notebooks in this repository.
