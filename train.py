import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_data(filename):
    """Load the dataset from a CSV file."""
    mileage = []
    price = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            mileage.append(float(row['km']))
            price.append(float(row['price']))
    
    return mileage, price

def normalize_data(data):
    """Normalize data to improve convergence of gradient descent."""
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val

def denormalize_theta(theta0, theta1, x_min, x_max, y_min, y_max):
    """Convert normalized theta values back to original scale."""
    denorm_theta1 = theta1 * (y_max - y_min) / (x_max - x_min)
    denorm_theta0 = y_min + (y_max - y_min) * theta0 - denorm_theta1 * x_min
    return denorm_theta0, denorm_theta1

def estimate_price(mileage, theta0, theta1):
    """Estimate price using the linear model."""
    return theta0 + (theta1 * mileage)

def compute_cost(mileage, price, theta0, theta1):
    """Compute the mean squared error cost function."""
    m = len(mileage)
    total_error = 0
    
    for i in range(m):
        prediction = estimate_price(mileage[i], theta0, theta1)
        total_error += (prediction - price[i]) ** 2
    
    return total_error / (2 * m)

def gradient_descent(mileage, price, learning_rate, iterations):
    """Perform gradient descent to find optimal theta values."""
    m = len(mileage)
    theta0 = 0
    theta1 = 0
    cost_history = []
    
    for _ in range(iterations):
        # Calculate temporary values for simultaneous update
        tmp_theta0 = learning_rate * (1/m) * sum(estimate_price(mileage[i], theta0, theta1) - price[i] for i in range(m))
        tmp_theta1 = learning_rate * (1/m) * sum((estimate_price(mileage[i], theta0, theta1) - price[i]) * mileage[i] for i in range(m))
        
        # Update theta values simultaneously
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        
        # Calculate and store cost for this iteration
        cost = compute_cost(mileage, price, theta0, theta1)
        cost_history.append(cost)
    
    return theta0, theta1, cost_history

def save_parameters(theta0, theta1):
    """Save the learned parameters to a file."""
    params = {
        'theta0': theta0,
        'theta1': theta1
    }
    
    with open('model_params.json', 'w') as file:
        json.dump(params, file)
    
    print(f"Parameters saved: theta0 = {theta0}, theta1 = {theta1}")

def plot_data_and_model(mileage, price, theta0, theta1, normalized_mileage=None, normalized_price=None):
    """Plot the data points and the fitted line."""
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(mileage, price, color='blue', label='Data points')
    
    # Plot the regression line
    x = np.linspace(min(mileage), max(mileage), 100)
    y = theta0 + theta1 * x
    plt.plot(x, y, color='red', label='Linear regression')
    
    plt.title('Car Price Prediction Model')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('regression_plot.png')
    
    # If normalized data is provided, plot it too
    if normalized_mileage and normalized_price:
        plt.figure(figsize=(10, 6))
        plt.scatter(normalized_mileage, normalized_price, color='green', label='Normalized data')
        x = np.linspace(0, 1, 100)
        y = theta0 + theta1 * x
        plt.plot(x, y, color='red', label='Linear regression')
        plt.title('Normalized Data and Model')
        plt.xlabel('Normalized Mileage')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.savefig('normalized_plot.png')
    
    plt.close('all')

def plot_cost_history(cost_history):
    """Plot the cost function over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
    plt.title('Cost Function over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.savefig('cost_history.png')
    plt.close()

def main():
    # Load data
    data_file = 'data.csv'
    mileage, price = load_data(data_file)
    
    # Normalize data
    norm_mileage, mileage_min, mileage_max = normalize_data(mileage)
    norm_price, price_min, price_max = normalize_data(price)
    
    # Set hyperparameters
    learning_rate = 0.1
    iterations = 1000
    
    # Train the model
    print("Training the model...")
    norm_theta0, norm_theta1, cost_history = gradient_descent(norm_mileage, norm_price, learning_rate, iterations)
    
    # Denormalize theta values
    theta0, theta1 = denormalize_theta(norm_theta0, norm_theta1, mileage_min, mileage_max, price_min, price_max)
    
    # Save the parameters
    save_parameters(theta0, theta1)
    
    # Plot the data and model
    plot_data_and_model(mileage, price, theta0, theta1, norm_mileage, norm_price)
    plot_cost_history(cost_history)
    
    print("Training complete!")
    print(f"Final cost: {cost_history[-1]}")
    print(f"Model: Price = {theta0:.2f} + {theta1:.2f} * mileage")

if __name__ == "__main__":
    main()
