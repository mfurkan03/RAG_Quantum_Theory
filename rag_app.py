import streamlit as st
import requests
import json

st.set_page_config(page_title="Document QA with RAG", page_icon="🧠")
st.title("📚 RAG-Based Document Assistant")
st.write("Sorgunuzu aşağıya yazın, sistem belgelerden size bilgi getirsin.")

# User input
user_query = st.text_input("Sorgunuz", placeholder="Örnek: 'Kuantum mekaniğinin temel ilkeleri nedir?'")

if st.button("Gönder") and user_query.strip() != "":
    with st.spinner("Yanıt getiriliyor..."):
        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"query": user_query}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display the response
                st.markdown("### Yanıt")
                st.write(result["response"])
                
                # Display sources
                st.markdown("### Kaynaklar")
                for i, in result["sources"]:
                    st.write(f"**Kaynak {i+1}:** {i['source']}, Sayfa: {i['page']}")
            else:
                st.error(f"Hata: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Backend ile iletişim hatası: {e}")