# Loan Predict AI – RAG-Powered Loan Eligibility Chatbot

An intelligent loan eligibility prediction system combining **Machine Learning** and **Retrieval-Augmented Generation (RAG)** for **Dream Housing Finance**.  
The system answers loan-related queries, predicts eligibility, and provides actionable insights through an interactive web interface.

---

## Features

- **RAG-Powered Q&A**: Intelligent chatbot for loan-related questions
- **ML Predictions**: Random Forest classifier for loan eligibility
- **Interactive Web UI**: Streamlit-based responsive interface
- **Batch Processing**: Upload CSV files for bulk predictions
- **Data Analytics**: Visual insights and approval trends
- **Real-time Chat**: Contextual responses with loan advice

---

## Tech Stack

- **Machine Learning**: Random Forest, Scikit-learn, Pandas  
- **NLP**: Sentence Transformers (`all-MiniLM-L6-v2`), FAISS  
- **Frontend**: Streamlit, Plotly  
- **Backend**: Python 3.10+

---

## Quick Setup

### 1️. Clone & Install
```bash
git clone <https://github.com/Harshit130127/RAG_Loan_Eligibility_Chatbot.git>
cd loan_predict_ai
pip install -r requirements.txt
```

### 2️. Add Dataset
```
dataset/
── Training_Dataset.csv
── Test_Dataset.csv
── Sample_Submission.csv
```

### 3️. Run Application
```bash
python run.py
```
App opens at: `http://localhost:8501`

---

##  Project Structure
```
dreamloan-ai/
├── src/
│   ├── app.py                # Streamlit web app
│   ├── model_training.py     # ML model training
│   ├── rag_chatbot.py        # RAG implementation
│   └── data_preprocessing.py # Data utilities
├── dataset/                  # CSV data files
├── models/                   # Trained models
├── requirements.txt
└── run.py                    # Main launcher
```

---

## Usage

###  Chat Interface
- Ask loan-related questions in natural language  
- Get instant eligibility predictions  
- Receive personalized loan advice  

###  Data Analysis
- Interactive charts and insights  
- Analyze loan approval patterns  
- Explore dataset statistics  

###  Batch Prediction
- Upload CSV for multiple predictions  
- Download results in submission format  
- Process large datasets efficiently  

---

##  Model Performance

- **Accuracy**: 85.2%  
- **Top Features**:  
  - Credit History (35%)  
  - Income (29%)  
  - Loan Amount (18%)  
- **RAG Model**: `all-MiniLM-L6-v2` 

---

##  Key Components

###  Machine Learning Pipeline
- Data preprocessing & missing value handling  
- Feature engineering (total income, EMI calculation)  
- Random Forest classification with cross-validation  

###  RAG System
- FAISS vector database for semantic search  
- Knowledge base with loan eligibility info  
- Context-aware response generation  

---

##  Example Queries

- "What factors affect loan approval?"  
- "How can I improve my chances?"  
- "Am I eligible for this loan?"  
- "What documents do I need?"  

---

##  Dataset

- **Records**: 615 loan applications  
- **Features**: Gender, Income, Credit History, Property Area, etc.  
- **Target**: Loan Status (Y/N)  




*Built with using Python, Streamlit & RAG Technology*
