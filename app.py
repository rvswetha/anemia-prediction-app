import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template, flash, redirect, url_for
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load model with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    logging.info("Model loaded successfully")
except FileNotFoundError:
    logging.error("Model file 'model.pkl' not found. Please run model.py first.")
    model = None
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    model = None

def validate_input_data(gender, hemoglobin, mch, mchc, mcv):
    """
    Validate input data for medical reasonableness
    
    Args:
        gender (float): Gender (0 or 1)
        hemoglobin (float): Hemoglobin level (g/dL)
        mch (float): Mean Corpuscular Hemoglobin (pg)
        mchc (float): Mean Corpuscular Hemoglobin Concentration (g/dL)
        mcv (float): Mean Corpuscular Volume (fL)
    
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    # Gender validation
    if gender not in [0, 1]:
        errors.append("Gender must be 0 (Male) or 1 (Female)")
    
    # Hemoglobin validation (typical range: 5-25 g/dL)
    if not (5.0 <= hemoglobin <= 25.0):
        errors.append("Hemoglobin must be between 5.0 and 25.0 g/dL")
    
    # MCH validation (typical range: 15-50 pg)
    if not (15.0 <= mch <= 50.0):
        errors.append("MCH must be between 15.0 and 50.0 pg")
    
    # MCHC validation (typical range: 20-40 g/dL)
    if not (20.0 <= mchc <= 40.0):
        errors.append("MCHC must be between 20.0 and 40.0 g/dL")
    
    # MCV validation (typical range: 50-120 fL)
    if not (50.0 <= mcv <= 120.0):
        errors.append("MCV must be between 50.0 and 120.0 fL")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""

@app.route('/')
def home():
    """
    Display the home page with prediction form
    
    Returns:
        str: Rendered HTML template
    """
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering home page: {str(e)}")
        return "An error occurred while loading the page.", 500

@app.route('/predict', methods=["POST"])
def predict():
    """
    Handle prediction requests with comprehensive error handling
    
    Returns:
        str: Rendered prediction results or error page
    """
    try:
        # Check if model is loaded
        if model is None:
            error_msg = "Prediction model is not available. Please contact support."
            logging.error("Prediction attempted but model is not loaded")
            return render_template("error.html", error_message=error_msg), 500
        
        # Extract and validate form data
        try:
            gender = float(request.form["Gender"])
            hemoglobin = float(request.form["Hemoglobin"])
            mch = float(request.form["MCH"])
            mchc = float(request.form["MCHC"])
            mcv = float(request.form["MCV"])
            
            logging.info(f"Prediction request: Gender={gender}, Hemoglobin={hemoglobin}, MCH={mch}, MCHC={mchc}, MCV={mcv}")
            
        except (ValueError, KeyError) as e:
            error_msg = "Invalid input data. Please enter valid numeric values for all fields."
            logging.error(f"Invalid input data: {str(e)}")
            return render_template("error.html", error_message=error_msg), 400
        
        # Validate medical ranges
        is_valid, validation_error = validate_input_data(gender, hemoglobin, mch, mchc, mcv)
        if not is_valid:
            logging.warning(f"Input validation failed: {validation_error}")
            return render_template("error.html", error_message=f"Input validation error: {validation_error}"), 400
        
        # Prepare data for prediction
        features_values = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
        
        # Make prediction
        try:
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)
            
            logging.info(f"Prediction made: {prediction[0]}")
            
        except Exception as e:
            error_msg = "Error occurred during prediction. Please try again."
            logging.error(f"Prediction error: {str(e)}")
            return render_template("error.html", error_message=error_msg), 500
        
        # Process results
        result = prediction[0]
        confidence = max(prediction_proba[0]) * 100  # Get confidence percentage
        
        if result == 0:
            result_text = "You don't have any Anemic Disease"
            result_class = "negative"
        elif result == 1:
            result_text = "You have anemic disease"
            result_class = "positive"
        else:
            error_msg = "Unexpected prediction result. Please contact support."
            logging.error(f"Unexpected prediction result: {result}")
            return render_template("error.html", error_message=error_msg), 500
        
        # Prepare response text
        prediction_text = f"Hence, based on calculation: {result_text} (Confidence: {confidence:.1f}%)"
        
        # Log successful prediction
        logging.info(f"Successful prediction: {result_text} with {confidence:.1f}% confidence")
        
        return render_template("predict.html", 
                             prediction_text=prediction_text,
                             result_class=result_class,
                             confidence=confidence)
        
    except Exception as e:
        # Catch any unexpected errors
        error_msg = "An unexpected error occurred. Please try again later."
        logging.error(f"Unexpected error in predict route: {str(e)}")
        return render_template("error.html", error_message=error_msg), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logging.warning("404 error occurred")
    return render_template("error.html", error_message="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logging.error("500 error occurred")
    return render_template("error.html", error_message="Internal server error occurred."), 500

if __name__ == "__main__":
    app.run(debug=False)