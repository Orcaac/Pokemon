import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fuzzywuzzy import process

# Define the path to your data file
data_path = r'C:\Users\Orcaaac\Downloads\pokemon_combined(2).csv'  # Update this path as needed

# Check if the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file at {data_path} does not exist. Please check the path.")

# Load Data from CSV
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for and handle missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Drop rows with missing values (or you can choose to fill them with a specific value)
data = data.dropna()

# Calculate Total Stats by summing the individual stats
data['Total_Stats'] = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].sum(axis=1)

# Rename 'Catch rate' column to 'Catch_Rate'
data.rename(columns={'Catch rate': 'Catch_Rate'}, inplace=True)

# Feature Engineering - Scaling Weight Based on Catch Rate
# Initialize the Min-Max Scaler
scaler = MinMaxScaler()

# Check for zero or extremely small values in Weight before scaling
data = data[data['Weight'] > 0]  # Remove rows with zero or negative weight

# Reshape data to scale it correctly (Scaler expects 2D input)
data['Weight_Scaled'] = scaler.fit_transform(data[['Weight']])

# Adding 'Movement_Speed' feature based on scaled weight (example formula)
data['Movement_Speed'] = 100 / data['Weight_Scaled']

# Check for and handle infinite or invalid values in Movement_Speed
data = data[~np.isinf(data['Movement_Speed'])]  # Remove rows with infinite values

# Step 3: Data Preparation
# Select features (Total_Stats and Movement_Speed) and the target variable (Catch_Rate)
X = data[['Total_Stats', 'Movement_Speed']]
y = data['Catch_Rate']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display the model's coefficients and intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Step 5: User Input to Select Pokémon
# Show available Pokémon and their stats
print("\nAvailable Pokémon:")
print(data[['Name', 'Total_Stats', 'Weight', 'Catch_Rate']])

# Allow the user to choose a Pokémon by name
chosen_pokemon_name = input("\nEnter the name of the Pokémon you want to select: ").capitalize()

# Check if the Pokémon name exists in the dataset
if chosen_pokemon_name in data['Name'].values:
    # Get the row corresponding to the chosen Pokémon
    chosen_pokemon = data[data['Name'] == chosen_pokemon_name].iloc[0]
    
    # Extract the necessary values
    chosen_total_stats = chosen_pokemon['Total_Stats']
    chosen_weight = chosen_pokemon['Weight']

    # Scale the weight using the fitted scaler
    chosen_weight_scaled = scaler.transform([[chosen_weight]])[0][0]
    if chosen_weight_scaled > 0:  # Check to avoid division by zero
        chosen_movement_speed = 100 / chosen_weight_scaled  # Calculate movement speed
    else:
        chosen_movement_speed = float('inf')  # Assign a large value if scaling is invalid

    # Create a DataFrame for the chosen Pokémon's data
    chosen_pokemon_data = pd.DataFrame({'Total_Stats': [chosen_total_stats], 'Movement_Speed': [chosen_movement_speed]})

    # Predict the catch rate for the chosen Pokémon
    predicted_catch_rate = model.predict(chosen_pokemon_data)

    # Limit the catch rate to a maximum of 100%
    predicted_catch_rate = min(predicted_catch_rate[0], 100)

    print(f"\nPredicted Catch Rate for {chosen_pokemon_name} with Total Stats {chosen_total_stats} and Weight {chosen_weight} kg: {predicted_catch_rate:.2f}%")
else:
    # Find the closest match if the Pokémon name is incorrect
    closest_match = process.extractOne(chosen_pokemon_name, data['Name'])
    if closest_match:
        print(f"The Pokémon name you entered does not exist in the dataset. Did you mean '{closest_match[0]}'?")
    else:
        print("No similar Pokémon names found in the dataset. Please check the spelling.")