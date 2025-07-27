import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import json

# Import path configuration for local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class LoanRAGChatbot:
    """
    RAG-based chatbot for loan eligibility assistance and prediction.
    
    This class implements a Retrieval-Augmented Generation (RAG) system
    for the Dream Housing Finance loan prediction project developed during
    my internship. It combines semantic search with machine learning predictions
    to provide intelligent, context-aware responses about loan eligibility.
    
    The RAG architecture ensures responses are grounded in factual information
    rather than hallucinated content, making it suitable for financial applications.
    """
    
    def __init__(self, loan_predictor):
        """
        Initialize the RAG chatbot with ML predictor and knowledge base.
        
        Args:
            loan_predictor: Trained machine learning model for loan predictions
        """
        self.loan_predictor = loan_predictor
        
        # Initialize sentence transformer for semantic embeddings
        # Using all-MiniLM-L6-v2 for balanced performance and model size
        # This model converts text to 384-dimensional vectors for similarity search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build the loan domain knowledge base
        self.knowledge_base = self._create_knowledge_base()
        
        # Initialize vector storage components
        self.embeddings = None
        self.index = None
        
        # Build FAISS vector store for efficient similarity search
        self._build_vector_store()
        
    def _create_knowledge_base(self):
        """
        Create curated knowledge base for loan eligibility domain.
        
        This knowledge base contains expert information about loan approval
        factors, documentation requirements, and application tips. During
        the internship, I researched common customer queries and banking
        practices to build this comprehensive database.
        
        In a production system, this would be loaded from external sources
        like databases, documents, or knowledge management systems.
        
        Returns:
            list: Structured knowledge base with question-answer pairs
        """
        knowledge = [
            {
                "question": "What factors affect loan approval?",
                "answer": "Loan approval depends on several factors: Credit History (most important), Total Income (Applicant + Coapplicant), Loan Amount, Employment status, Property Area, Education level, and Marital Status. Credit history has the highest impact on approval."
            },
            {
                "question": "How important is credit history?",
                "answer": "Credit history is the most crucial factor for loan approval. Applicants with good credit history (value 1) have significantly higher chances of approval compared to those with poor credit history (value 0)."
            },
            {
                "question": "Does income affect loan approval?",
                "answer": "Yes, higher total income (applicant + coapplicant income) increases approval chances. The bank considers the debt-to-income ratio and ensures the applicant can afford the EMI."
            },
            {
                "question": "What is the role of property area?",
                "answer": "Property area affects loan approval. Urban properties generally have higher approval rates, followed by semi-urban, then rural areas. This is due to property value and market stability considerations."
            },
            {
                "question": "How does education level impact approval?",
                "answer": "Graduate applicants typically have higher approval rates than non-graduates, as education level often correlates with income stability and job security."
            },
            {
                "question": "Does marital status matter?",
                "answer": "Married applicants often have higher approval rates as they may have dual income sources (coapplicant income) and are perceived as more financially stable."
            },
            {
                "question": "What about self-employment?",
                "answer": "Self-employed applicants may face slightly lower approval rates due to income variability, but it's not a major deciding factor if other criteria are met."
            },
            {
                "question": "How is loan amount determined?",
                "answer": "Loan amount should be reasonable compared to income. Very high loan amounts relative to income reduce approval chances. The EMI should not exceed 40-50% of total income."
            },
            {
                "question": "What documents are required?",
                "answer": "Typical documents include: Income proof (salary slips, ITR), Identity proof (Aadhar, PAN), Address proof, Bank statements, Property documents, and Credit report."
            },
            {
                "question": "How to improve loan approval chances?",
                "answer": "To improve approval chances: Maintain good credit score, increase total income (add coapplicant), reduce existing debts, choose appropriate loan amount, and ensure all documents are complete and accurate."
            }
        ]
        return knowledge
    
    def _build_vector_store(self):
        """
        Build FAISS vector store for efficient similarity search.
        
        This method implements the core retrieval component of the RAG architecture.
        It converts the knowledge base into searchable vector embeddings and creates
        a FAISS index for fast similarity search operations.
        
        The vector store enables semantic search over our loan knowledge base,
        allowing the chatbot to find relevant information even when users
        phrase questions differently from the stored questions.
        """
        # Convert knowledge base Q&A pairs to searchable text format
        texts = []
        for item in self.knowledge_base:
            # Combine question and answer for richer context during retrieval
            texts.append(f"Question: {item['question']} Answer: {item['answer']}")
        
        # Generate embeddings using sentence transformers
        # This converts text to numerical vectors for similarity computation
        self.embeddings = self.encoder.encode(texts)
        
        # Build FAISS index for efficient similarity search
        # Using IndexFlatIP for inner product similarity (equivalent to cosine after normalization)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity computation
        # This ensures similarity scores are between -1 and 1
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
    
    def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant documents from knowledge base using semantic search.
        
        This implements the retrieval step of the RAG pipeline. It converts the
        user query to an embedding vector and finds the most similar documents
        in the knowledge base using FAISS vector search.
        
        Args:
            query: User's question or search term
            top_k: Number of most similar documents to retrieve
            
        Returns:
            List of relevant Q&A context strings
        """
        # Convert user query to embedding vector for similarity search
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search using FAISS
        # Returns similarity scores and indices of most similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Extract relevant context from knowledge base
        relevant_contexts = []
        for idx in indices[0]:
            if idx < len(self.knowledge_base):
                context = self.knowledge_base[idx]
                # Format context as Q&A pairs for better readability
                relevant_contexts.append(f"Q: {context['question']}\nA: {context['answer']}")
        
        return relevant_contexts
    
    def predict_loan_eligibility(self, applicant_data: Dict) -> Dict:
        """
        Predict loan eligibility using the trained machine learning model.
        
        This method integrates the ML prediction component with the RAG chatbot,
        allowing users to get instant loan eligibility assessments through
        natural conversation. It processes applicant data and returns
        structured prediction results with confidence scores.
        
        Args:
            applicant_data: Dictionary containing loan application details
            
        Returns:
            Dictionary with loan status, probability, and confidence scores
        """
        try:
            # Convert applicant data to pandas DataFrame for ML model input
            df = pd.DataFrame([applicant_data])
            
            # Generate prediction using the trained Random Forest model
            predictions, probabilities = self.loan_predictor.predict(df)
            
            # Structure prediction results for API response
            result = {
                'loan_status': predictions[0],  # Y/N decision
                'approval_probability': float(probabilities[0][1]) if predictions[0] == 'Y' else float(probabilities[0][0]),
                'confidence': max(float(probabilities[0][1]), float(probabilities[0][0]))  # Max probability as confidence
            }
            
            return result
        except Exception as e:
            # Handle any prediction errors gracefully
            return {'error': str(e)}
    
    def chat(self, user_query: str, applicant_data: Dict = None) -> str:
        """
        Main chat function implementing RAG (Retrieval-Augmented Generation) architecture.
        
        This method processes user queries and generates contextually appropriate responses
        by combining retrieved knowledge with machine learning predictions. It implements
        the complete RAG pipeline: Retrieval → Augmentation → Generation.
        
        The function handles two main types of interactions:
        1. Prediction requests: Uses ML model to assess loan eligibility
        2. Information queries: Uses RAG to provide knowledge-based responses
        
        Args:
            user_query: Natural language question from user
            applicant_data: Optional loan application data for predictions
            
        Returns:
            Generated response with relevant loan information
        """
        user_query_lower = user_query.lower()
        
        # Check if user is requesting loan prediction
        # Look for prediction-related keywords in the query
        if any(keyword in user_query_lower for keyword in ['predict', 'eligible', 'approval', 'loan status']):
            if applicant_data:
                # Generate ML prediction with applicant data
                result = self.predict_loan_eligibility(applicant_data)
                
                if 'error' in result:
                    return f"Sorry, I couldn't process your loan application: {result['error']}"
                
                # Format prediction results for user-friendly display
                status = "approved" if result['loan_status'] == 'Y' else "rejected"
                confidence = result['confidence'] * 100
                
                # Create comprehensive response with key factors analysis
                response = f"""
Based on your application details, your loan is likely to be **{status}**.

**Confidence:** {confidence:.1f}%

**Key Factors Analysis:**
- Credit History: {'Good' if applicant_data.get('Credit_History', 0) == 1 else 'Poor'} (Most Important)
- Total Income: ₹{applicant_data.get('ApplicantIncome', 0) + applicant_data.get('CoapplicantIncome', 0):,}
- Loan Amount: ₹{applicant_data.get('LoanAmount', 0) * 1000:,}
- Property Area: {applicant_data.get('Property_Area', 'Unknown')}

Would you like tips on improving your approval chances?
"""
                return response
            else:
                # Guide user to provide required information for prediction
                return "To predict loan eligibility, please provide your application details including income, credit history, loan amount, etc."
        
        # RAG Pipeline: Retrieve relevant context from knowledge base
        relevant_contexts = self._retrieve_relevant_context(user_query, top_k=2)
        
        # RAG Pipeline: Generate response based on retrieved context
        if relevant_contexts:
            # Augmentation step: Combine retrieved context with user query
            context = "\n\n".join(relevant_contexts)
            
            # Generation step: Create contextual response using retrieved information
            response = f"""Based on the loan eligibility knowledge:

{context}

Is there anything specific about loan approval you'd like to know more about?"""
        else:
            # Fallback response when no relevant context is found
            # Provide helpful guidance about available capabilities
            response = """I can help you with:
1. **Loan Eligibility Prediction** - Provide your details for instant prediction
2. **Loan Approval Factors** - What affects your chances
3. **Tips for Improvement** - How to increase approval probability
4. **Documentation** - Required documents
5. **Process Questions** - General loan process queries

What would you like to know?"""
        
        return response
