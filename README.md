# Linear Regression Car Price Predictor

This project implements a simple linear regression model to predict car prices based on mileage. It serves as an introduction to machine learning concepts, specifically focusing on gradient descent optimization.

## Project Structure

- `train.py`: Trains the linear regression model using gradient descent
- `predict.py`: Uses the trained model to predict car prices for given mileage
- `evaluate.py`: Calculates and displays precision metrics for the model
- `data.csv`: Sample dataset with car mileage and price data
- `model_params.json`: Generated after training, contains the model parameters
- `regression_plot.png`: Visualization of data points and the fitted line
- `normalized_plot.png`: Visualization of normalized data and model
- `cost_history.png`: Graph showing cost decrease during training
- `residual_plot.png`: Plot showing prediction errors across mileage values

## Mathematical Model

The model uses a simple linear equation to predict car prices:

```
estimatePrice(mileage) = θ0 + (θ1 * mileage)
```

Where:
- θ0 is the y-intercept (base price)
- θ1 is the slope (price change per km)

## How to Use

### Requirements

- Python 3.x
- matplotlib
- numpy

Install the required packages:

```
pip install matplotlib numpy
```

### Training the Model

Run the training script to learn the model parameters:

```
python train.py
```

This will:
1. Load the data from `data.csv`
2. Normalize the data for better convergence
3. Apply gradient descent to find optimal θ0 and θ1 values
4. Save the parameters to `model_params.json`
5. Generate visualization plots

### Predicting Prices

After training, use the prediction script to estimate car prices:

```
python predict.py
```

Simply enter a mileage value when prompted, and the program will return the estimated price.

### Evaluating the Model

To assess the precision of the model:

```
python evaluate.py
```

This will display various metrics including R-squared, Mean Squared Error, and Mean Absolute Error, along with sample predictions compared to actual values.

## Detailed Explanation of the Algorithm

### 1. Linear Regression Model

The core of this project is a simple linear regression model that assumes a linear relationship between car mileage (x) and price (y):

```
y = θ0 + θ1 * x
```

- **θ0 (theta0)**: The y-intercept, representing the base price of a car with 0 km
- **θ1 (theta1)**: The slope, representing how much the price changes per kilometer

### 2. Data Normalization

Before training, we normalize the data to improve the convergence of gradient descent:

```python
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val
```

This scales all values to be between 0 and 1, which helps the gradient descent algorithm converge more efficiently. After training, we denormalize the parameters to get back to the original scale:

```python
def denormalize_theta(theta0, theta1, x_min, x_max, y_min, y_max):
    denorm_theta1 = theta1 * (y_max - y_min) / (x_max - x_min)
    denorm_theta0 = y_min + (y_max - y_min) * theta0 - denorm_theta1 * x_min
    return denorm_theta0, denorm_theta1
```

### 3. Cost Function

To measure how well our model fits the data, we use the Mean Squared Error (MSE) cost function:

```python
def compute_cost(mileage, price, theta0, theta1):
    m = len(mileage)
    total_error = 0
    
    for i in range(m):
        prediction = estimate_price(mileage[i], theta0, theta1)
        total_error += (prediction - price[i]) ** 2
    
    return total_error / (2 * m)
```

This calculates the average squared difference between predicted and actual prices. The factor of 1/2 is a convention that makes the derivative calculations cleaner.

### 4. Gradient Descent Algorithm

Gradient descent is an iterative optimization algorithm that finds the values of θ0 and θ1 that minimize the cost function:

```python
def gradient_descent(mileage, price, learning_rate, iterations):
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
```

For each iteration:
1. We calculate the gradient (direction of steepest increase) of the cost function with respect to each parameter
2. We update the parameters in the opposite direction of the gradient (to minimize cost)
3. The learning rate controls the step size of each update
4. We track the cost at each iteration to monitor convergence

The update formulas are derived from calculus (partial derivatives of the cost function):

```
tmpθ0 = learningRate * (1/m) * sum(estimatePrice(mileage[i]) - price[i])
tmpθ1 = learningRate * (1/m) * sum((estimatePrice(mileage[i]) - price[i]) * mileage[i])
```

### 5. Visualization Generation

#### Regression Plot

The regression plot shows the original data points and the fitted line:

```python
def plot_data_and_model(mileage, price, theta0, theta1):
    plt.figure(figsize=(10, 6))
    
    # Plot original data points
    plt.scatter(mileage, price, color='blue', label='Data points')
    
    # Plot the regression line
    x = np.linspace(min(mileage), max(mileage), 100)
    y = theta0 + theta1 * x
    plt.plot(x, y, color='red', label='Linear regression')
```

- **Data points**: Each point (x, y) represents a car with mileage x and price y
- **Regression line**: Generated by calculating y = θ0 + θ1 * x for 100 evenly spaced x values between the minimum and maximum mileage

#### Cost History Plot

The cost history plot shows how the cost function decreases during training:

```python
def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history)
```

Each point (i, cost) shows the value of the cost function after i iterations of gradient descent. A decreasing trend indicates that the algorithm is converging.

#### Residual Plot

The residual plot shows the difference between actual and predicted values:

```python
def plot_residuals(mileage, price, theta0, theta1):
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    residuals = [p - pred for p, pred in zip(price, predictions)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, residuals, color='purple')
    plt.axhline(y=0, color='red', linestyle='-')
```

For each data point:
1. We calculate the predicted price using our model
2. We compute the residual (actual price - predicted price)
3. We plot the residual against the mileage
4. The horizontal red line at y=0 represents perfect predictions

### 6. Model Evaluation Metrics

#### R-squared (Coefficient of Determination)

R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variable:

```python
def calculate_r_squared(mileage, price, theta0, theta1):
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    mean_price = sum(price) / len(price)
    ss_total = sum((p - mean_price) ** 2 for p in price)
    ss_residual = sum((p - pred) ** 2 for p, pred in zip(price, predictions))
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared
```

- **SS Total**: Sum of squared differences between actual prices and the mean price (total variance)
- **SS Residual**: Sum of squared differences between actual and predicted prices (unexplained variance)
- **R-squared**: 1 - (SS Residual / SS Total), ranges from 0 to 1, with 1 indicating a perfect fit

#### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values:

```python
def calculate_mse(mileage, price, theta0, theta1):
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    mse = sum((p - pred) ** 2 for p, pred in zip(price, predictions)) / len(price)
    return mse
```

#### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual values:

```python
def calculate_mae(mileage, price, theta0, theta1):
    predictions = [estimate_price(m, theta0, theta1) for m in mileage]
    mae = sum(abs(p - pred) for p, pred in zip(price, predictions)) / len(price)
    return mae
```

## Results

After training on the provided dataset, the model achieves:
- θ0 = 8481.17 (base price for a car with 0 km)
- θ1 = -0.02 (price decreases by about 2 cents per kilometer)
- R-squared = 0.7329 (73.29% of price variance explained by mileage)
- Mean Absolute Error = $556.50 (average prediction error)

## Conclusion

This simple linear regression model demonstrates the fundamental concepts of machine learning:
1. Defining a hypothesis (linear model)
2. Defining a cost function (MSE)
3. Optimizing parameters using gradient descent
4. Evaluating model performance

While more complex models might achieve better accuracy, this linear approach provides a solid foundation for understanding machine learning algorithms.
