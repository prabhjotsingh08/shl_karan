"""Streamlit chatbot UI for the SHL assessment recommender."""

from __future__ import annotations

import os
from typing import List

import requests
import streamlit as st


API_URL = os.environ.get("RECOMMENDER_API_URL", "http://localhost:8000")


def fetch_recommendations(query: str) -> List[dict]:
    """Fetch recommendations from the API."""
    response = requests.post(f"{API_URL}/recommend", json={"query": query}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("recommended_assessments", [])


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(page_title="SHL Assessment Recommender", layout="wide", page_icon="💼")

# Header
st.title("💼 SHL Assessment Recommendation Chatbot")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #1f2937;'>
            👋 <strong>Welcome!</strong> I can help you find the perfect SHL assessments for your hiring needs. 
            Simply describe the role, job requirements, or what you're looking to assess, and I'll recommend 
            the best individual assessments for you.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Describe the job role or assessment needs..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get recommendations
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your requirements and finding the best assessments..."):
            try:
                assessments = fetch_recommendations(prompt)
                
                if not assessments:
                    response_text = (
                        "I couldn't find any assessments matching your requirements. "
                        "Could you try providing more details about the role or what you're looking to assess?"
                    )
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    # Success message
                    st.success(f"✅ Found {len(assessments)} recommended assessment(s) for you!")
                    
                    # Display each recommendation
                    for idx, assessment in enumerate(assessments, 1):
                        with st.expander(
                            f"📋 **{idx}. {assessment.get('name', 'Unknown Assessment')}**",
                            expanded=(idx == 1)  # Expand first one by default
                        ):
                            # Description
                            if assessment.get("description"):
                                st.markdown(f"**Description:** {assessment['description']}")
                            
                            # Test types
                            if assessment.get("test_type"):
                                types_str = ", ".join(assessment["test_type"])
                                st.markdown(f"**Test Type:** {types_str}")
                            
                            # Duration
                            if assessment.get("duration"):
                                st.markdown(f"**Duration:** {assessment['duration']} minutes")
                            
                            # Remote support
                            if assessment.get("remote_support"):
                                remote_icon = "✅" if assessment["remote_support"] == "Yes" else "❌"
                                st.markdown(f"**Remote Testing:** {remote_icon} {assessment['remote_support']}")
                            
                            # Adaptive support
                            if assessment.get("adaptive_support"):
                                adaptive_icon = "✅" if assessment["adaptive_support"] == "Yes" else "❌"
                                st.markdown(f"**Adaptive:** {adaptive_icon} {assessment['adaptive_support']}")
                            
                            # URL
                            if assessment.get("url"):
                                st.markdown(f"**🔗 [View Assessment Details]({assessment['url']})**")
                    
                    # Build response text for chat history
                    response_text = f"I found {len(assessments)} recommended assessment(s) for you:\n\n"
                    for idx, assessment in enumerate(assessments, 1):
                        response_text += f"{idx}. **{assessment.get('name', 'Unknown')}**\n"
                        if assessment.get("description"):
                            response_text += f"   - {assessment['description'][:100]}...\n"
                        if assessment.get("url"):
                            response_text += f"   - [View Details]({assessment['url']})\n\n"
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
            except requests.RequestException as exc:
                error_msg = f"❌ Sorry, I encountered an error while fetching recommendations: {str(exc)}\n\nPlease make sure the backend API is running at {API_URL}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as exc:
                error_msg = f"❌ An unexpected error occurred: {str(exc)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This chatbot helps you find the right SHL individual assessments 
        based on your job requirements or hiring needs.
        
        **How to use:**
        1. Describe the role or what you want to assess
        2. I'll analyze your requirements
        3. Receive personalized assessment recommendations
        
        **Example queries:**
        - "I need assessments for a software engineer role"
        - "Looking for cognitive ability tests for entry-level positions"
        - "Personality assessments for management roles"
        """
    )
    
    st.header("🔧 Settings")
    api_url = st.text_input("API URL", value=API_URL, help="Backend API endpoint")
    if api_url != API_URL:
        os.environ["RECOMMENDER_API_URL"] = api_url
        st.rerun()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
