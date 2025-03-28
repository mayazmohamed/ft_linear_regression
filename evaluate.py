import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from train import load_data

def load_parameters():
    """Load the trained model parameters."""
    try:
        with open('model_params.json', 'r') as file:
            params = json.load(file)
        return params['theta0'], params['theta1']
    except FileNotFoundError:
        print("Error: model_params.json not found. Please run train.py first.")
        return 0, 0

def estimate_price(mileage, theta0, theta1):
    """Estimate price using the linear model."""
    return theta0 + (theta1 * mileage)

def calculate_r_squared(mileage, price, theta0, theta1):
    """Calculate the R-squared (coefficient of determination)."""
    # Calculate predictions
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    
    # Calculate mean of actual prices
    mean_price = sum(price) / len(price)
    
    # Calculate total sum of squares (proportional to variance of data)
    ss_total = sum((p - mean_price) ** 2 for p in price)
    
    # Calculate sum of squares of residuals
    ss_residual = sum((p - pred) ** 2 for p, pred in zip(price, predictions))
    
    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total)
    
    return r_squared

def calculate_mse(mileage, price, theta0, theta1):
    """Calculate Mean Squared Error."""
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    mse = sum((p - pred) ** 2 for p, pred in zip(price, predictions)) / len(price)
    return mse

def calculate_mae(mileage, price, theta0, theta1):
    """Calculate Mean Absolute Error."""
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    mae = sum(abs(p - pred) for p, pred in zip(price, predictions)) / len(price)
    return mae

def plot_residuals(mileage, price, theta0, theta1):
    """Plot the residuals to visualize prediction errors."""
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    residuals = [p - pred for p, pred in zip(price, predictions)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.grid(True)
    plt.savefig('residual_plot.png')
    plt.close()

def main():
    # Load data and parameters
    data_file = 'data.csv'
    mileage, price = load_data(data_file)
    theta0, theta1 = load_parameters()
    
    if theta0 == 0 and theta1 == 0:
        return
    
    # Calculate metrics
    r_squared = calculate_r_squared(mileage, price, theta0, theta1)
    mse = calculate_mse(mileage, price, theta0, theta1)
    mae = calculate_mae(mileage, price, theta0, theta1)
    
    # Plot residuals
    plot_residuals(mileage, price, theta0, theta1)
    
    # Print results
    print("\nModel Evaluation Metrics")
    print("------------------------")
    print(f"Model: Price = {theta0:.2f} + ({theta1:.2f} * mileage)")
    print(f"R-squared: {r_squared:.4f} (Higher is better, 1.0 is perfect fit)")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
    print("------------------------")
    
    # Calculate and display predictions vs actual values
    print("\nSample Predictions vs Actual Values:")
    print("Mileage    | Actual Price | Predicted Price | Difference")
    print("-" * 60)
    
    # Show a few sample predictions
    samples = min(10, len(mileage))  # Show up to 10 samples
    indices = np.linspace(0, len(mileage)-1, samples, dtype=int)
    
    for i in indices:
        actual = price[i]
        predicted = estimate_price(mileage[i], theta0, theta1)
        diff = actual - predicted
        print(f"{mileage[i]:10.0f} | ${actual:11.2f} | ${predicted:14.2f} | ${diff:10.2f}")
    
    print("\nResidual plot saved as 'residual_plot.png'")

if __name__ == "__main__":
    main()
