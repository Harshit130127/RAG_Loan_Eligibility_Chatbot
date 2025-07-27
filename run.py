"""
Main script to run the Loan Eligibility RAG Chatbot Application
"""

import os
import sys
import pandas as pd
from src.model_training import LoanPredictor

def setup_project():
    """Setup project directories and train model if needed"""
    # Creating directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    # Check if model exists, if not train it
    model_path = 'models/loan_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        
        # Check if training data exists
        training_data_paths = [
            'dataset/Training_Dataset.csv',     
            'dataset/Training-Dataset.csv',      
            'Training_Dataset.csv',              
            'Training-Dataset.csv'               
        ]
        
        training_data_path = None
        for path in training_data_paths:
            if os.path.exists(path):
                training_data_path = path
                break
        
        if not training_data_path:
            print("Error: Training dataset not found!")
            print("Please ensure the training dataset is in one of these locations:")
            for path in training_data_paths:
                print(f"  - {path}")
            print("\nCurrent directory contents:")
            if os.path.exists('dataset'):
                print("Dataset folder contents:")
                for file in os.listdir('dataset'):
                    print(f"  - dataset/{file}")
            else:
                print("Dataset folder not found!")
            return False
        
        # Train model
        predictor = LoanPredictor()
        train_data = pd.read_csv(training_data_path)
        print(f"Loading training data from: {training_data_path}")
        
        accuracy = predictor.train_model(train_data)
        predictor.save_model(model_path)
        
        print(f"Model trained successfully with accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
    else:
        print("Pre-trained model found. Loading existing model...")
    
    # Verify test dataset exists
    test_data_paths = [
        'dataset/Test_Dataset.csv',
        'dataset/Test-Dataset.csv',
        'Test_Dataset.csv',
        'Test-Dataset.csv'
    ]
    
    test_data_found = False
    for path in test_data_paths:
        if os.path.exists(path):
            print(f"Test dataset found at: {path}")
            test_data_found = True
            break
    
    if not test_data_found:
        print("Warning: Test dataset not found. You can upload files manually in the web interface.")
        print("Looking for test data in these locations:")
        for path in test_data_paths:
            print(f"  - {path}")
    
    return True

def main():
    """Main function to run the application"""
    print("Dream Housing Finance - Loan Eligibility RAG Chatbot")
    print("=" * 60)
    
    # Checking current directory structure
    print("Current directory structure:")
    print(f"Current working directory: {os.getcwd()}")
    
    if os.path.exists('dataset'):
        print("Dataset folder contents:")
        for file in os.listdir('dataset'):
            print(f"  - {file}")
    else:
        print("Dataset folder not found!")
    
    print("-" * 60)
    
    # Setup project
    if not setup_project():
        print("\nProject setup failed. Please check your data files and try again.")
        return
    
    print("\nStarting Streamlit application...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nTo stop the application, press Ctrl+C")
    print("=" * 60)
    
    # Run streamlit app
    os.system("streamlit run src/app.py")

if __name__ == "__main__":
    main()
