import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Data preprocessing class for loan eligibility prediction.
    
    This class handles the complete data preprocessing pipeline including
    missing value imputation, categorical encoding, and feature engineering
    required for the machine learning model in this internship project.
    """
    
    def __init__(self):
        """
        Initialize the DataPreprocessor with empty label encoders dictionary.
        
        The label_encoders will store fitted encoders during training phase
        and reuse them for consistent encoding during prediction phase.
        """
        self.label_encoders = {}
        
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the loan application dataset for machine learning.
        
        This method implements the complete data preprocessing pipeline
        developed for the Dream Housing Finance loan prediction system.
        
        Args:
            df (pandas.DataFrame): Raw loan application data
            is_training (bool): Flag to indicate training vs prediction mode
            
        Returns:
            pandas.DataFrame: Preprocessed data ready for ML model
        """
        # Create a copy to avoid modifying the original dataset
        df = df.copy()
        
        # Handle missing values using domain-appropriate strategies
        # For categorical variables, use mode (most frequent value)
        # For numerical variables, use median to handle outliers
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        
        # Convert categorical variables to numerical format for ML algorithms
        # Machine learning models require numerical input, so we encode text categories
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                          'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            if is_training:
                # During training: fit new label encoder and store for future use
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # During prediction: use previously fitted encoders for consistency
                if col in self.label_encoders:
                    # Handle unseen categories that weren't in training data
                    # This prevents errors when new categorical values appear in test data
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        # Replace unseen categories with the most frequent training category
                        # This ensures the model can handle new categorical values gracefully
                        most_frequent = self.label_encoders[col].classes_[0]
                        df[col] = df[col].astype(str).replace(list(unseen_values), most_frequent)
                    
                    # Apply the fitted encoder to maintain consistency with training
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Feature engineering: Create new meaningful features for better prediction
        # These derived features help the model understand financial relationships
        
        # Total household income combines applicant and coapplicant income
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Estimated Monthly Installment (EMI) calculation
        # This represents the monthly payment burden for the loan
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
        
        # Balance income after EMI payment
        # This indicates the remaining income after loan payment (multiply EMI by 1000 for proper scaling)
        df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)
        
        return df
