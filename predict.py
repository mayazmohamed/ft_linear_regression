import json
import os

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

def main():
    # Load the trained parameters
    theta0, theta1 = load_parameters()
    
    if theta0 == 0 and theta1 == 0:
        return
    
    print("Car Price Prediction Model")
    print("--------------------------")
    print(f"Model: Price = {theta0:.2f} + ({theta1:.2f} * mileage)")
    print("--------------------------")
    
    while True:
        try:
            mileage_input = input("\nEnter car mileage (km) or 'q' to quit: ")
            
            if mileage_input.lower() == 'q':
                break
                
            mileage = float(mileage_input)
            if mileage < 0:
                print("Error: Mileage cannot be negative.")
                continue
                
            price = estimate_price(mileage, theta0, theta1)
            print(f"Estimated price: ${price:.2f}")
            
        except ValueError:
            print("Error: Please enter a valid number for mileage.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
