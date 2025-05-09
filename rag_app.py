import streamlit as st
import requests
import json

st.set_page_config(page_title="Document QA with RAG", page_icon="🧠")
st.title("⚛️ RAG-Based Document Assistant")
st.write("Enter your question below, the system will bring information form the database!")

# User input
user_query = st.text_input("Question", placeholder="Example: 'What is quantum theory?'")

if st.button("Send!") and user_query.strip() != "":
    with st.spinner("Generating the answer..."):
        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": user_query}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display the response
                st.markdown("### Answer")
                st.write(result["response"])
                
                # Display sources
                st.markdown("### Sources")
                for i,j in enumerate(result["sources"]):
                    st.write(f"**Kaynak {i+1}:** {j['source']}, Sayfa: {j['page']}")
            else:
                st.error(f"ERROR: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error while communicating with backend: {e}")