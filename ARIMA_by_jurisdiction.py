# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# File path
file_path = 'physicians-in-canada-1971-2022.xlsx'

# Specify the range of columns and rows to read
columns_range = 'A:BH'
num_rows = 101698

# Read the specific range from the Excel file into a DataFrame
df = pd.read_excel(file_path, sheet_name='Table 1 Physician workforce', usecols=columns_range, nrows=num_rows)

# Replace NaN values in Health region column with the corresponding 'Jurisdiction' values to simplify the filtering by province
df['Health region'] = df.apply(
    lambda row: row['Jurisdiction'] if pd.isna(row['Health region']) else row['Health region'],
    axis=1
)

# Define a list of jurisdictions
jurisdictions = [
    'Canada', 'Alta.', 'B.C.', 'Man.', 'N.B.', 'N.L.', 
    'N.S.', 'N.W.T.', 'Ont.', 'P.E.I.', 'Que.', 'Sask.', 'Y.T.','Nun.'
]

# Create a dictionary to store DataFrames for each jurisdiction
jurisdiction_dfs = {}

# Loop through each jurisdiction and create the DataFrame
for jurisdiction in jurisdictions:
    jurisdiction_key = jurisdiction.replace('.', '').replace(' ', '_').lower()
    jurisdiction_dfs[jurisdiction_key] = df[
        (df['Specialty sort'] == 3) &
        ((df['Jurisdiction'] == jurisdiction) & (df['Health region'] == jurisdiction))
    ]

# Define a function to evaluate ARIMA model to get the best parameters for each jurisdiction (province)
def evaluate_arima_model(train, validation, arima_order):
    try:
        model = ARIMA(train, order=arima_order)
        model_fit = model.fit()
        y_pred = model_fit.get_forecast(steps=len(validation))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df["Predictions"] = model_fit.predict(start=validation.index[0], end=validation.index[-1])
        y_pred_df.index = validation.index
        y_pred_out = y_pred_df["Predictions"]
        rmse = np.sqrt(mean_squared_error(validation.values, y_pred_out))
        return rmse
    except Exception as e:
        print(f"Failed to fit ARIMA{arima_order}: {str(e)}")
        return float('inf')  # Return infinite RMSE if model fails to converge

# Create a dictionary to store the best models for each jurisdiction (province)
best_models = {}

# Loop through each jurisdiction (province) and prepare data for ARIMA
for jurisdiction, df in jurisdiction_dfs.items():
    try:
        # Extract 'Number of physicians' and 'Year' columns and sort by 'Year'
        jurisdiction_data = df[['Year', 'Number of physicians']].sort_values(by='Year').reset_index(drop=True)
        
        # Split data into train and validation sets
        train_size = int(len(jurisdiction_data) * 0.8)
        train, validation = jurisdiction_data['Number of physicians'][:train_size], jurisdiction_data['Number of physicians'][train_size:]
        
        # Grid search for ARIMA hyperparameters
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)

        best_score, best_cfg = float("inf"), None

        for p, d, q in itertools.product(p_values, d_values, q_values):
            order = (p, d, q)
            try:
                rmse = evaluate_arima_model(train, validation, order)
                if rmse < best_score:
                    best_score, best_cfg = rmse, order
            except Exception as e:
                print(f"Issue with ARIMA{order} for {jurisdiction}: {str(e)}")
                continue
        
        # Store and print the best model result for each jurisdiction (province)
        best_models[jurisdiction] = (best_cfg, best_score)
        #print(f'Best ARIMA model for {jurisdiction}: ARIMA{best_cfg} RMSE={best_score}')
        
    except Exception as e:
        print(f"Error processing jurisdiction {jurisdiction}: {str(e)}")
        continue

# Bring back the ARIMA model function for making predictions and plotting
def evaluate_arima_model(train, validation, arima_order):
    try:
        model = ARIMA(train, order=arima_order)
        model_fit = model.fit()
        y_pred = model_fit.get_forecast(steps=len(validation))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df["Predictions"] = model_fit.predict(start=validation.index[0], end=validation.index[-1])
        y_pred_df.index = validation.index
        y_pred_out = y_pred_df["Predictions"]
        rmse = np.sqrt(mean_squared_error(validation.values, y_pred_out))
        return rmse, model_fit, y_pred_df
    except Exception as e:
        print(f"Failed to fit ARIMA{arima_order}: {str(e)}")
        return float('inf'), None, None  # Return infinite RMSE if model fails to converge

# Loop through each jurisdiction (province) and prepare data for ARIMA
for jurisdiction, df in jurisdiction_dfs.items():
    try:
        # Extract 'Year' and 'Number of physicians' columns and sort by 'Year'
        jurisdiction_data = df[['Year', 'Number of physicians']].sort_values(by='Year').reset_index(drop=True)
        
        # Split data into train and validation sets
        train_size = int(len(jurisdiction_data) * 0.8)
        train, validation = jurisdiction_data['Number of physicians'][:train_size], jurisdiction_data['Number of physicians'][train_size:]
        
        # Use the best ARIMA model parameters for the jurisdiction
        arima_order = best_models[jurisdiction][0]  # Assuming best_models contains the best (p, d, q) for each jurisdiction
        
        # Evaluate the best ARIMA model on the validation set
        rmse, model_fit, y_pred_df = evaluate_arima_model(train, validation, arima_order)
        
        # Print the best ARIMA model results for each jurisdiction
        print(f'Best ARIMA model for {jurisdiction}: ARIMA{arima_order} RMSE={rmse}')
        
        if model_fit is None:
            print(f'Failed to find a suitable ARIMA model for {jurisdiction}')
            continue
        
        # Plotting for the best ARIMA model
        if model_fit is not None:
            # Make predictions on the validation set
            y_pred_out = y_pred_df["Predictions"]
            
            # Save the training and validation plot
            plt.figure(figsize=(12, 6))
            plt.plot(train, label='Training Data')
            plt.plot(validation, label='Validation Data')
            plt.plot(y_pred_out, color='yellow', label='ARIMA Predictions')
            plt.title(f'ARIMA Predictions for {jurisdiction} (ARIMA{arima_order})')
            plt.legend()
            plt.savefig(f'{jurisdiction}_arima_predictions.png')
            plt.close()
            
            # Fit model on the entire dataset
            ARIMAmodel_full = ARIMA(jurisdiction_data['Number of physicians'], order=arima_order)
            ARIMAmodel_full = ARIMAmodel_full.fit()
            
            # Forecast future values (10 years)
            forecast_steps = 10  # Number of future periods to forecast
            future_forecast = ARIMAmodel_full.get_forecast(steps=forecast_steps)
            future_forecast_df = future_forecast.conf_int(alpha=0.05)
            future_forecast_df["Predictions"] = ARIMAmodel_full.predict(start=len(jurisdiction_data), end=len(jurisdiction_data) + forecast_steps - 1)
            
            # Save the future forecast plot
            plt.figure(figsize=(12, 6))
            plt.plot(jurisdiction_data['Year'], jurisdiction_data['Number of physicians'], label='Historical Data')
            future_years = range(jurisdiction_data['Year'].max() + 1, jurisdiction_data['Year'].max() + forecast_steps + 1)  # Add 1 to include all forecast years
            plt.plot(future_years, future_forecast_df["Predictions"], color='yellow', label='Future Forecast')
            plt.title(f'Future Forecast for {jurisdiction} (ARIMA{arima_order})')
            plt.legend()
            plt.savefig(f'{jurisdiction}_future_forecast.png')
            plt.close()

    except Exception as e:
        print(f"Error processing jurisdiction {jurisdiction}: {str(e)}")
        continue
