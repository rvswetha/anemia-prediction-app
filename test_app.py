import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory to the path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import pickle

class TestAnemiaApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and test data"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Load the model for testing
        try:
            self.model = pickle.load(open('model.pkl', 'rb'))
        except FileNotFoundError:
            self.model = None
    
    def test_home_page(self):
        """Test if home page loads successfully"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Anemia Prediction', response.data)
    
    def test_valid_prediction_no_anemia(self):
        """Test prediction with valid data that should indicate no anemia"""
        response = self.app.post('/predict', data={
            'Gender': '0',      # Male
            'Hemoglobin': '15.2',  # High hemoglobin
            'MCH': '30.5',
            'MCHC': '34.1',
            'MCV': '88.3'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"don't have", response.data)
    
    def test_valid_prediction_anemia(self):
        """Test prediction with valid data that should indicate anemia"""
        response = self.app.post('/predict', data={
            'Gender': '1',      # Female
            'Hemoglobin': '8.5',   # Low hemoglobin
            'MCH': '22.1',
            'MCHC': '28.9',
            'MCV': '78.2'
        })
        self.assertEqual(response.status_code, 200)
        # Should indicate anemia (not contain "don't have")
    
    def test_invalid_data_empty_fields(self):
        """Test prediction with empty fields"""
        response = self.app.post('/predict', data={
            'Gender': '',
            'Hemoglobin': '',
            'MCH': '',
            'MCHC': '',
            'MCV': ''
        })
        # Should handle this gracefully (we'll add error handling)
        self.assertEqual(response.status_code, 500)
    
    def test_invalid_data_non_numeric(self):
        """Test prediction with non-numeric data"""
        response = self.app.post('/predict', data={
            'Gender': 'invalid',
            'Hemoglobin': 'not_a_number',
            'MCH': '22.1',
            'MCHC': '28.9',
            'MCV': '78.2'
        })
        # Should handle this gracefully (we'll add error handling)
        self.assertEqual(response.status_code, 500)
    
    def test_model_exists(self):
        """Test if model file exists and can be loaded"""
        self.assertIsNotNone(self.model, "Model file should exist and be loadable")
    
    def test_model_prediction_format(self):
        """Test if model returns expected prediction format"""
        if self.model:
            # Test with sample data
            sample_data = pd.DataFrame([[0, 12.5, 28.0, 32.0, 85.0]], 
                                     columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
            prediction = self.model.predict(sample_data)
            
            # Should return array with single value (0 or 1)
            self.assertEqual(len(prediction), 1)
            self.assertIn(prediction[0], [0, 1])
    
    def test_boundary_values(self):
        """Test prediction with boundary values"""
        # Test with extreme low values
        response = self.app.post('/predict', data={
            'Gender': '0',
            'Hemoglobin': '5.0',   # Very low
            'MCH': '15.0',         # Very low
            'MCHC': '25.0',        # Low
            'MCV': '60.0'          # Low
        })
        self.assertEqual(response.status_code, 200)
        
        # Test with extreme high values
        response = self.app.post('/predict', data={
            'Gender': '1',
            'Hemoglobin': '20.0',  # Very high
            'MCH': '40.0',         # Very high
            'MCHC': '40.0',        # High
            'MCV': '110.0'         # High
        })
        self.assertEqual(response.status_code, 200)

class TestDataValidation(unittest.TestCase):
    """Test data validation functions (to be implemented)"""
    
    def test_hemoglobin_range(self):
        """Test if hemoglobin values are within reasonable medical range"""
        # Normal range: 12-15.5 g/dL for women, 13.5-17.5 g/dL for men
        # We should flag values outside 5-25 g/dL as potentially erroneous
        valid_values = [8.5, 12.0, 15.5, 18.0]
        invalid_values = [-1, 0, 30, 100]
        
        # This will be implemented when we add validation functions
        pass
    
    def test_gender_values(self):
        """Test if gender values are valid (0 or 1)"""
        valid_values = [0, 1, '0', '1']
        invalid_values = [2, -1, 'male', 'female', '']
        
        # This will be implemented when we add validation functions
        pass

if __name__ == '__main__':
    unittest.main()