import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Import path configuration for local modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import DataPreprocessor

class LoanPredictor:
    """
    Machine Learning model for loan eligibility prediction.
    
    This class implements the complete ML pipeline for the Dream Housing Finance
    loan prediction system developed during my internship. It handles model
    training, validation, prediction, and persistence operations.
    """
    
    def __init__(self):
        """
        Initialize the LoanPredictor with empty model and preprocessor.
        
        The model and preprocessor will be populated during training phase
        and used together to ensure consistent data processing pipeline.
        """
        self.model = None
        self.preprocessor = None
        
    def train_model(self, train_data):
        """
        Train the Random Forest model for loan eligibility prediction.
        
        This method implements the complete machine learning pipeline including
        data preprocessing, model training, and validation. I chose Random Forest
        over other algorithms because it handles categorical features well and
        provides feature importance insights for loan decision explanation.
        
        Args:
            train_data (pandas.DataFrame): Training dataset with loan applications
            
        Returns:
            float: Validation accuracy score for model performance assessment
        """
        
        # Initialize and apply data preprocessing pipeline
        # The preprocessor handles missing values, categorical encoding, and feature engineering
        self.preprocessor = DataPreprocessor()
        processed_data = self.preprocessor.preprocess_data(train_data, is_training=True)
        
        # Separate features from target variable for supervised learning
        # Drop non-predictive columns (Loan_ID is just an identifier)
        X = processed_data.drop(['Loan_ID', 'Loan_Status'], axis=1)
        
        # Convert categorical target (Y/N) to numerical format (1/0) for ML algorithms
        y = processed_data['Loan_Status'].map({'Y': 1, 'N': 0})
        
        # Split data into training and validation sets for model evaluation
        # Using 80-20 split with fixed random state for reproducible results
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train Random Forest classifier
        # n_estimators=100 provides good balance between performance and training time
        # random_state ensures reproducible results for internship demonstration
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Validate model performance on held-out validation set
        # This gives us an unbiased estimate of model performance
        val_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, val_pred)
        
        # Display validation metrics for performance assessment
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_pred))
        
        return accuracy
    
    def predict(self, test_data):
        """
        Generate loan eligibility predictions for new applications.
        
        This method applies the trained model to make predictions on new loan
        applications. It uses the same preprocessing pipeline as training to
        ensure data consistency and reliable predictions.
        
        Args:
            test_data (pandas.DataFrame): New loan applications for prediction
            
        Returns:
            tuple: (prediction_labels, prediction_probabilities)
                - prediction_labels: List of 'Y'/'N' loan decisions
                - prediction_probabilities: Array of prediction confidence scores
                
        Raises:
            ValueError: If model hasn't been trained yet
        """
        # Verify that model has been trained before making predictions
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained yet!")
        
        # Apply the same preprocessing pipeline used during training
        # is_training=False ensures consistent encoding without refitting encoders
        processed_data = self.preprocessor.preprocess_data(test_data, is_training=False)
        
        # Remove identifier column to match training feature set
        X_test = processed_data.drop(['Loan_ID'], axis=1)
        
        # Generate predictions using the trained Random Forest model
        predictions = self.model.predict(X_test)
        pred_proba = self.model.predict_proba(X_test)
        
        # Convert numerical predictions (1/0) back to business format (Y/N)
        # This matches the original data format expected by the business
        pred_labels = ['Y' if pred == 1 else 'N' for pred in predictions]
        
        return pred_labels, pred_proba
    
    def save_model(self, filepath):
        """
        Persist the trained model and preprocessor to disk.
        
        This method saves both the trained ML model and the fitted preprocessor
        to ensure the complete pipeline can be reused for future predictions
        without retraining. Essential for production deployment.
        
        Args:
            filepath (str): Path where the model should be saved
        """
        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save both model and preprocessor together to maintain pipeline integrity
        # Using pickle for efficient serialization of scikit-learn objects
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'preprocessor': self.preprocessor}, f)
    
    def load_model(self, filepath):
        """
        Load a previously trained model and preprocessor from disk.
        
        This method restores the complete ML pipeline including both the trained
        model and the fitted preprocessor. This enables making predictions
        without retraining, which is crucial for production applications.
        
        Args:
            filepath (str): Path to the saved model file
        """
        # Load the saved model and preprocessor from pickle file
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.model = saved_data['model']
            self.preprocessor = saved_data['preprocessor']
