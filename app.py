from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('car_price.pk1' , 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    car_brand = request.form['car_brand']
    kms_driven = float(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']
    age = int(request.form['age'])
    
    # Map car brand and fuel type to numerical features
    # These mappings should align with your training data preprocessing
    car_brand_score = map_car_brand(car_brand)  # Function for mapping brands
    fuel_one_hot = one_hot_encode_fuel(fuel_type)  # e.g., [1, 0, 0] for Petrol
    
    # Combine features
    features = [car_brand_score, kms_driven] + fuel_one_hot + [age]
    features = np.array([features])  # Reshape to 2D array
    
    # Predict the price
    predicted_price = model.predict(features)[0]
    
    return render_template('result.html', prediction=predicted_price)

def map_car_brand(brand):
    car_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Toyota', 'Volkswagen', 'Porsche', 'Volvo', 'Jaguar', 'Land',
                  'Ford', 'Honda', 'Chevrolet', 'Hyundai', 'Kia', 'Nissan', 'Renault', 'Skoda', 'MINI', 'MG', 'Mahindra',
                  'Tata', 'Isuzu', 'Jeep', 'Datsun', 'Fiat', 'Citroen', 'Bentley', 'Maruti']
    try:
        # Apply the same encoding logic
        encoded_value = ((car_brands.index(brand) + 1) / (1874)) * 100
    except ValueError:
        # Handle unknown brands with a default value (e.g., 0)
        encoded_value = 0
    return encoded_value

def one_hot_encode_fuel(fuel):
    # Replace with your one-hot encoding logic
    fuel_mapping = {'Petrol': [1, 0, 0], 'Diesel': [0, 1, 0], 'CNG': [0, 0, 1]}
    return fuel_mapping.get(fuel, [0, 0, 0])

if __name__ == '__main__':
    app.run(debug=True)
