import streamlit as st
import requests
from fastapi import FastAPI

app = FastAPI()
# Backend sunucunun URL'si (yerelde çalışıyorsan localhost:8000)
BACKEND_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Document QA with RAG", page_icon="🧠")
st.title("📚 RAG-Based Document Assistant")
st.write("Sorgunuzu aşağıya yazın, sistem belgelerden size bilgi getirsin.")

# Kullanıcıdan sorgu al
user_query = st.text_input("Sorgunuz", placeholder="Örnek: 'Kuantum mekaniğinin temel ilkeleri nedir?'")

if st.button("Gönder") and user_query.strip() != "":
    with st.spinner("Yanıt getiriliyor..."):
        try:
            response = requests.post(
                BACKEND_URL,
                json={"query": user_query},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Yanıt:")
                st.markdown(result["answer"])
                if "sources" in result:
                    with st.expander("🔍 Kaynaklar"):
                        for i, chunk in enumerate(result["sources"], 1):
                            st.markdown(f"**Parça {i}:**\n```\n{chunk}\n```")
            else:
                st.error("Bir hata oluştu: " + response.text)
        except Exception as e:
            st.error(f"İstek gönderilirken hata oluştu: {e}")

@app.post("/query")
def handle_query(request: str):
    # 1. Embed the query
    # 2. Retrieve relevant chunks from ChromaDB
    # 3. Concatenate query and context
    # 4. Send to LLM (e.g., Mistral 7B or Ollama)
    # 5. Return the generated answer and optionally the source chunks
    return {"answer", "sources"}