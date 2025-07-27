import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import path configuration for local modules
# This ensures Python can find our custom modules in the src directory
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom machine learning and chatbot modules
from model_training import LoanPredictor
from rag_chatbot import LoanRAGChatbot

# Streamlit page configuration
# Setting up the web application's basic properties for professional appearance
st.set_page_config(
    page_title="Dream Housing Finance - Loan Assistant",
    layout="wide"  # Using wide layout for better data visualization and user experience
)

# Custom CSS styling for the application
# Creating a professional look with proper color schemes and responsive design
st.markdown("""
<style>
    /* Main header styling for the application title */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chat message container styling for better readability */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000 !important;  /* Ensuring readability with black text */
    }
    
    /* User message styling with blue theme for distinction */
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000000 !important;
    }
    
    /* Bot message styling with green theme for distinction */
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        color: #000000 !important;
    }
    
    /* Styling for user and bot labels in chat */
    .user-message strong {
        color: #1565c0 !important;
    }
    .bot-message strong {
        color: #2e7d32 !important;
    }
    
    /* Making all buttons full width for better UX */
    .stButton > button {
        width: 100%;
    }
    
    /* Custom metric container styling for professional appearance */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        color: #000000 !important;
    }
    
    /* White text styling for specific UI elements */
    .white-text {
        color: #ffffff !important;
    }
    
    /* Main content text styling */
    .main-content-text {
        color: #ffffff !important;
    }
    
    /* Sidebar text styling */
    .sidebar-text {
        color: #ffffff !important;
    }
    
    /* Form input labels styling for better visibility */
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label {
        color: #ffffff !important;
    }
    
    /* Instruction text styling */
    .instruction-text {
        color: #ffffff !important;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model loading and caching function
# Using Streamlit's caching to avoid reloading the model on every interaction
@st.cache_resource
def load_model():
    """
    Load or train the machine learning model with caching for performance optimization.
    This function implements the ML pipeline for loan eligibility prediction.
    """
    model_path = 'models/loan_model.pkl'  # Path where trained model is stored
    predictor = LoanPredictor()  # Initialize our custom predictor class
    
    # Try to load existing trained model first to save time
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
            return predictor
        except Exception as e:
            # If loading fails, we'll train a new model
            st.warning(f"Error loading model: {e}. Training new model...")
    
    # Model training process
    # This runs only when no trained model exists or loading fails
    with st.spinner(" Training model for the first time... This may take a few minutes."):
        # Define possible paths for training data to handle different file structures
        training_paths = [
            'dataset/Training_Dataset.csv',      # Primary expected path
            'dataset/Training-Dataset.csv',      # Alternative naming convention
            'Training_Dataset.csv',              # Root directory fallback
            'Training-Dataset.csv'               # Alternative root naming
        ]
        
        # Initialize variables for data loading
        train_data = None
        training_path_used = None
        
        # Attempt to load training data from various possible locations
        for path in training_paths:
            if os.path.exists(path):
                try:
                    train_data = pd.read_csv(path)
                    training_path_used = path
                    print(f" Loaded training data from: {path}")
                    break
                except Exception as e:
                    print(f"❌ Error loading {path}: {e}")
                    continue
        
        # Error handling for missing training data
        if train_data is None:
            st.error("❌ Training data not found! Please check if Training_Dataset.csv exists in the dataset folder.")
            st.stop()
        
        # Train the model and handle any errors
        try:
            accuracy = predictor.train_model(train_data)
            predictor.save_model(model_path)  # Save trained model for future use
            st.success(f"✅ Model trained successfully using data from: {training_path_used}")
            st.info(f" Model Accuracy: {accuracy:.4f}")
            return predictor
        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            st.stop()

# Session state initialization
# Using Streamlit's session state to maintain application state across interactions

# Initialize the chatbot only once per session
if 'chatbot' not in st.session_state:
    predictor = load_model()  # Load our trained model
    st.session_state.chatbot = LoanRAGChatbot(predictor)  # Initialize RAG chatbot with predictor

# Initialize chat history to store conversation
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize test data loading state for batch processing
if 'test_data_loaded' not in st.session_state:
    st.session_state.test_data_loaded = None

# Application header with professional styling
st.markdown('<h1 class="main-header"> Dream Housing Finance - AI Loan Assistant</h1>', unsafe_allow_html=True)

# Sidebar form for loan application
# Creating an interactive form for users to input their loan application details
st.sidebar.header(" Loan Application Form")
st.sidebar.write("Fill in your details for instant loan eligibility check:")

# Loan application form with all required fields
with st.sidebar.form("loan_form"):
    # Basic applicant information
    loan_id = st.text_input("Loan ID", value="LP999999", help="Enter unique loan application ID")
    
    # Organizing form fields in columns for better layout and space utilization
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    # Financial details section
    st.subheader(" Financial Details")
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=1000,
                                     help="Monthly income of primary applicant")
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000,
                                       help="Monthly income of co-applicant (if any)")
    loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=1, value=100, step=10,
                                help="Requested loan amount in thousands")
    loan_term = st.selectbox("Loan Amount Term (months)", [360, 180, 240, 300, 480],
                           help="Repayment period in months")
    credit_history = st.selectbox("Credit History", [1, 0], 
                                format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)",
                                help="1 = Good credit history, 0 = Poor credit history")
    
    # Form submission button
    submitted = st.form_submit_button(" Check Eligibility", use_container_width=True)

# Main application tabs
# Organizing the application into logical sections for better user experience
tab1, tab2, tab3 = st.tabs(["💬Chat Assistant", " Data Analysis", " Batch Prediction"])

# Tab 1: Chat Assistant - RAG-powered Q&A interface
with tab1:
    st.header(" AI Loan Assistant")
    st.write("Ask me anything about loan approval, requirements, or get instant eligibility predictions!")
    
    # Display chat history
    # Creating a container to show previous conversations
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                # Display user messages with distinct styling
                st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                # Display bot responses with distinct styling
                st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input interface
    user_input = st.text_input("Type your question here:", key="chat_input", 
                              placeholder="e.g., What factors affect loan approval?")
    
    # Chat control buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        # Send button functionality
        if st.button(" Send", key="send_chat", use_container_width=True):
            if user_input.strip():
                # Add user message to chat history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input
                })
                
                # Prepare applicant data if form was submitted
                applicant_data = None
                if submitted:
                    applicant_data = {
                        'Loan_ID': loan_id,
                        'Gender': gender,
                        'Married': married,
                        'Dependents': dependents,
                        'Education': education,
                        'Self_Employed': self_employed,
                        'ApplicantIncome': applicant_income,
                        'CoapplicantIncome': coapplicant_income,
                        'LoanAmount': loan_amount,
                        'Loan_Amount_Term': loan_term,
                        'Credit_History': credit_history,
                        'Property_Area': property_area
                    }
                
                # Generate response using RAG chatbot
                try:
                    response = st.session_state.chatbot.chat(user_input, applicant_data)
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response
                    })
                    
                    # Refresh the page to show new messages
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
            else:
                st.warning(" Please enter a message before sending.")
    
    with col2:
        # Clear chat functionality
        if st.button(" Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Quick action buttons for common queries
    st.subheader("⚡ Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Approval Factors", use_container_width=True):
            response = st.session_state.chatbot.chat("What factors affect loan approval?")
            st.session_state.chat_history.extend([
                {'type': 'user', 'content': 'What factors affect loan approval?'},
                {'type': 'bot', 'content': response}
            ])
            st.experimental_rerun()
    
    with col2:
        if st.button(" Improve Chances", use_container_width=True):
            response = st.session_state.chatbot.chat("How to improve loan approval chances?")
            st.session_state.chat_history.extend([
                {'type': 'user', 'content': 'How to improve loan approval chances?'},
                {'type': 'bot', 'content': response}
            ])
            st.experimental_rerun()
    
    with col3:
        if st.button(" Required Documents", use_container_width=True):
            response = st.session_state.chatbot.chat("What documents are required?")
            st.session_state.chat_history.extend([
                {'type': 'user', 'content': 'What documents are required?'},
                {'type': 'bot', 'content': response}
            ])
            st.experimental_rerun()

# Tab 2: Data Analysis - Comprehensive data visualization and insights
with tab2:
    st.header(" Loan Data Analysis")
    
    # Function to load training data with caching for performance
    @st.cache_data
    def load_training_data():
        """
        Load training data from various possible locations.
        Returns the dataframe and the path from which it was loaded.
        """
        training_data_paths = [
            'dataset/Training_Dataset.csv',
            'dataset/Training-Dataset.csv', 
            'Training_Dataset.csv',
            'Training-Dataset.csv'
        ]
        
        # Try each path until we find the data
        for path in training_data_paths:
            if os.path.exists(path):
                try:
                    return pd.read_csv(path), path
                except Exception as e:
                    continue
        return None, None
    
    # Load and validate training data
    df, data_path = load_training_data()
    
    if df is not None:
        st.success(f"✅ Training data loaded from: {data_path}")
        
        # Data quality overview section
        st.subheader(" Data Quality Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Display key data quality metrics
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col3:
            approved_count = (df['Loan_Status'] == 'Y').sum()
            st.metric("Approved Loans", approved_count)
        with col4:
            rejection_rate = ((df['Loan_Status'] == 'N').sum() / len(df)) * 100
            st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
        
        # Data visualization section
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit history impact visualization
            try:
                fig1 = px.histogram(df, x='Loan_Status', color='Credit_History', 
                                  title='Loan Approval by Credit History',
                                  color_discrete_map={1: 'green', 0: 'red'},
                                  labels={'Loan_Status': 'Loan Status', 'count': 'Number of Applications'})
                fig1.update_layout(showlegend=True, legend_title="Credit History")
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating credit history chart: {e}")
            
            # Income distribution analysis
            try:
                fig2 = px.box(df, y='ApplicantIncome', x='Loan_Status', 
                             title='Income Distribution by Loan Status',
                             labels={'ApplicantIncome': 'Applicant Income (₹)', 'Loan_Status': 'Loan Status'})
                fig2.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating income distribution chart: {e}")
        
        with col2:
            # Property area impact analysis
            try:
                area_approval = df.groupby(['Property_Area', 'Loan_Status']).size().unstack(fill_value=0)
                area_approval_pct = area_approval.div(area_approval.sum(axis=1), axis=0) * 100
                
                fig3 = px.bar(area_approval_pct, title='Approval Rate by Property Area (%)',
                             labels={'value': 'Percentage', 'Property_Area': 'Property Area'})
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating property area chart: {e}")
            
            # Education level distribution
            try:
                edu_approval = df.groupby(['Education', 'Loan_Status']).size().unstack(fill_value=0)
                
                fig4 = px.pie(values=edu_approval.sum(axis=1), names=edu_approval.index, 
                             title='Applications by Education Level')
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating education chart: {e}")
        
        # Key statistics summary
        st.subheader(" Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Display important metrics in styled containers
        with col1:
            total_apps = len(df)
            st.markdown(f'<div class="metric-container"><h3>{total_apps:,}</h3><p>Total Applications</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
            st.markdown(f'<div class="metric-container"><h3>{approval_rate:.1f}%</h3><p>Overall Approval Rate</p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            avg_income = df['ApplicantIncome'].mean()
            st.markdown(f'<div class="metric-container"><h3>₹{avg_income:,.0f}</h3><p>Average Income</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            avg_loan = df['LoanAmount'].mean()
            st.markdown(f'<div class="metric-container"><h3>₹{avg_loan*1000:,.0f}</h3><p>Average Loan Amount</p></div>', 
                       unsafe_allow_html=True)
        
        # Additional insights section
        st.subheader(" Additional Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            # Loan amount vs income correlation analysis
            try:
                df_clean = df.dropna(subset=['LoanAmount', 'ApplicantIncome'])
                df_clean = df_clean[df_clean['ApplicantIncome'] > 0]
                df_clean['Loan_to_Income_Ratio'] = (df_clean['LoanAmount'] * 1000) / df_clean['ApplicantIncome']
                
                fig5 = px.scatter(df_clean, x='ApplicantIncome', y='LoanAmount', 
                                 color='Loan_Status', title='Loan Amount vs Applicant Income',
                                 labels={'ApplicantIncome': 'Applicant Income (₹)', 
                                        'LoanAmount': 'Loan Amount (₹ thousands)'})
                st.plotly_chart(fig5, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating scatter plot: {e}")
        
        with col2:
            # Credit history impact quantification
            try:
                credit_impact = df.groupby('Credit_History')['Loan_Status'].apply(lambda x: (x == 'Y').mean() * 100)
                
                fig6 = px.bar(x=credit_impact.index, y=credit_impact.values,
                             title='Approval Rate by Credit History',
                             labels={'x': 'Credit History (0=Poor, 1=Good)', 'y': 'Approval Rate (%)'})
                st.plotly_chart(fig6, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating credit impact chart: {e}")
        
    else:
        # Error handling for missing training data
        st.error("❌ Training data not found for analysis!")
        st.info("Please ensure your training data file exists in one of these locations:")
        st.code("""
        - dataset/Training_Dataset.csv
        - dataset/Training-Dataset.csv  
        - Training_Dataset.csv
        - Training-Dataset.csv
        """)

# Tab 3: Batch Prediction - Process multiple loan applications
with tab3:
    st.header(" Batch Loan Prediction")
    
    # Create two columns for different input methods
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Automatic test dataset loading
        st.subheader(" Load Test Dataset")
        if st.button(" Load Test Dataset Automatically", use_container_width=True):
            # Define possible locations for test data
            test_data_paths = [
                'dataset/Test_Dataset.csv',
                'dataset/Test-Dataset.csv',
                'Test_Dataset.csv', 
                'Test-Dataset.csv'
            ]
            
            test_df = None
            test_path_used = None
            
            # Try to load test data from various locations
            for path in test_data_paths:
                if os.path.exists(path):
                    try:
                        test_df = pd.read_csv(path)
                        test_path_used = path
                        break
                    except Exception as e:
                        st.warning(f"Error loading {path}: {e}")
                        continue
            
            # Display results of data loading
            if test_df is not None:
                st.session_state.test_data_loaded = test_df
                st.success(f"✅ Test dataset loaded from: {test_path_used}")
                st.write(" Preview of test data:")
                st.dataframe(test_df.head(10))
                
                st.info(f"Dataset Info: {len(test_df)} records, {len(test_df.columns)} columns")
                
            else:
                st.error("❌ Test dataset not found!")
                st.info("Please ensure your test data file exists in one of these locations:")
                st.code("""
                - dataset/Test_Dataset.csv
                - dataset/Test-Dataset.csv
                - Test_Dataset.csv
                - Test-Dataset.csv
                """)
    
    with col2:
        # Manual file upload option
        st.subheader(" Upload Custom File")
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'],
                                        help="Upload a CSV file with the same format as training data")
        
        # Handle uploaded file
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.session_state.test_data_loaded = test_df
                st.success(f"✅ Custom file uploaded successfully!")
                st.write(" Preview of uploaded data:")
                st.dataframe(test_df.head(10))
                
                st.info(f" Dataset Info: {len(test_df)} records, {len(test_df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {e}")
    
    # Batch prediction processing
    if st.session_state.test_data_loaded is not None:
        st.divider()
        
        # Center the prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(" Generate Predictions", use_container_width=True, type="primary"):
                with st.spinner("Generating predictions... Please wait."):
                    try:
                        test_df = st.session_state.test_data_loaded
                        # Generate predictions using our trained model
                        predictions, probabilities = st.session_state.chatbot.loan_predictor.predict(test_df)
                        
                        # Create comprehensive results dataframe
                        results_df = test_df.copy()
                        results_df['Predicted_Loan_Status'] = predictions
                        results_df['Approval_Probability'] = [
                            prob[1] if pred == 'Y' else prob[0] 
                            for pred, prob in zip(predictions, probabilities)
                        ]
                        results_df['Confidence_Score'] = [
                            max(prob[0], prob[1]) 
                            for prob in probabilities
                        ]
                        
                        st.success("✅ Predictions generated successfully!")
                        
                        # Display prediction results
                        st.subheader(" Prediction Results")
                        
                        # Summary metrics for batch predictions
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            approval_count = sum(1 for pred in predictions if pred == 'Y')
                            st.metric("Approved Applications", approval_count)
                        
                        with col2:
                            rejection_count = len(predictions) - approval_count
                            st.metric("Rejected Applications", rejection_count)
                        
                        with col3:
                            approval_rate = (approval_count / len(predictions)) * 100
                            st.metric("Approval Rate", f"{approval_rate:.1f}%")
                        
                        with col4:
                            avg_confidence = results_df['Confidence_Score'].mean() * 100
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        
                        # Display detailed results table
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download options for results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Full results download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Full Results CSV",
                                data=csv,
                                file_name="loan_predictions_full.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Submission format download (matching competition requirements)
                            submission_df = results_df[['Loan_ID', 'Predicted_Loan_Status']].copy()
                            submission_df.columns = ['Loan_ID', 'Loan_Status']
                            submission_csv = submission_df.to_csv(index=False)
                            
                            st.download_button(
                                label=" Download Submission Format CSV",
                                data=submission_csv,
                                file_name="loan_submission.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        # Visualization of batch prediction results
                        st.subheader(" Prediction Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Prediction distribution pie chart
                            pred_counts = pd.Series(predictions).value_counts()
                            fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                       title="Prediction Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence score distribution histogram
                            fig2 = px.histogram(results_df, x='Confidence_Score',
                                              title="Confidence Score Distribution",
                                              labels={'Confidence_Score': 'Confidence Score'})
                            st.plotly_chart(fig2, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating predictions: {e}")
                        st.info("Please check if your data format matches the training data structure.")

# Form submission handling
# Process individual loan application when form is submitted
if submitted:
    # Prepare applicant data dictionary
    applicant_data = {
        'Loan_ID': loan_id,
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    try:
        # Generate prediction using our trained model
        result = st.session_state.chatbot.predict_loan_eligibility(applicant_data)
        
        if 'error' not in result:
            # Process successful prediction
            status = "✅ APPROVED" if result['loan_status'] == 'Y' else "❌ REJECTED"
            confidence = result['confidence'] * 100
            
            # Enhanced sidebar display with prediction results
            st.sidebar.markdown("---")
            st.sidebar.subheader(" Prediction Result")
            
            # Display approval/rejection status with appropriate styling
            if result['loan_status'] == 'Y':
                st.sidebar.success(f"**Status:** {status}")
            else:
                st.sidebar.error(f"**Status:** {status}")
            
            st.sidebar.info(f"**Confidence:** {confidence:.1f}%")
            
            # Calculate and display additional insights
            total_income = applicant_income + coapplicant_income
            loan_to_income_ratio = (loan_amount * 1000) / total_income if total_income > 0 else 0
            
            st.sidebar.write("**Key Factors:**")
            st.sidebar.write(f"• Credit History: {'Good ' if credit_history == 1 else 'Poor ❌'}")
            st.sidebar.write(f"• Total Income: ₹{total_income:,}")
            st.sidebar.write(f"• Loan Amount: ₹{loan_amount * 1000:,}")
            st.sidebar.write(f"• L/I Ratio: {loan_to_income_ratio:.1f}")
            
            # Add prediction result to chat history for context
            prediction_message = f"Loan prediction for {loan_id}: {status} (Confidence: {confidence:.1f}%)"
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': prediction_message
            })
            
        else:
            # Handle prediction errors
            st.sidebar.error(f"❌ Error: {result['error']}")
            
    except Exception as e:
        # Handle any unexpected errors during prediction
        st.sidebar.error(f"❌ Prediction Error: {e}")

# Application footer
# Professional footer with project information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
         Dream Housing Finance - AI Loan Assistant | 
        Built with python using Streamlit & RAG Technology
    </div>
    """, 
    unsafe_allow_html=True
)
