import pdfplumber
import os
from langchain.text_splitter import TokenTextSplitter
from langchain.schema.document import Document
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys
import contextlib
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import argparse
import requests
import streamlit as st
from fastapi import FastAPI
import torch 

# FastAPI app (we'll leave this for potential backend integration)
app = FastAPI()
BACKEND_URL = "http://localhost:8000/query"


def initialize_model():

    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return generator,tokenizer


@app.post("/query")
def handle_query(query: str,db_encoder,collection,generator,tokenizer):
    # 1. Embed the query
    # 2. Retrieve relevant chunks from ChromaDB
    # 3. Concatenate query and context
    # 4. Send to LLM (e.g., Mistral 7B or Ollama)
    # 5. Return the generated answer and optionally the source chunks

    with torch.no_grad():
        query_embedding = db_encoder.encode(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    context = " ".join(results["documents"][0])
    prompt = f"Given the context, please answer the question: {context}\n\n So, considering that my question is: {query}\n You can give your answer here:"
    chat_prompt = tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True)

    response = generator(chat_prompt, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    response = response.split("[/INST]")[1]

    return {"response": response, "sources": results["metadatas"][0]}



def main():
    """
    Main function to handle command-line interface and model initialization.
    """
    parser = argparse.ArgumentParser(description="RAG-Based Document Assistant")
    parser.add_argument(
        "-db_path", "--database_path", 
        type=str, 
        required=True, 
        help="Path that you put your database"
    )
    parser.add_argument(
        "-db_name", "--database_name", 
        type=str, 
        required=True, 
        help="Name of the database"
    )

    args = parser.parse_args()

    print(f"Initializing model: ...")
    
    # Initialize model with the provided argument
    generator,tokenizer = initialize_model()

    print(f"Retrieving data: ...")

    chroma_client = chromadb.PersistentClient(path=args.database_path)

    collection = chroma_client.get_collection(name=args.database_name)
    if len(collection.get(limit=1))>= 1:

        print(f"Database initialized successfully!")
        return generator,tokenizer,collection

if __name__ == "__main__":
    # If running from terminal, initialize the model first
    generator,tokenizer,collection = main()

    # Now start Streamlit
    st.set_page_config(page_title="Document QA with RAG", page_icon="🧠")
    st.title("📚 RAG-Based Document Assistant")
    st.write("Sorgunuzu aşağıya yazın, sistem belgelerden size bilgi getirsin.")
    
    # Kullanıcıdan sorgu al
    user_query = st.text_input("Sorgunuz", placeholder="Örnek: 'Kuantum mekaniğinin temel ilkeleri nedir?'")
    db_encoder = SentenceTransformer('all-MiniLM-L6-v2',device = "cuda")

    if st.button("Gönder") and user_query.strip() != "":
        with st.spinner("Yanıt getiriliyor..."):
            handle_query(user_query,db_encoder,collection,generator,tokenizer)