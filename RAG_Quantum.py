
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import argparse
import streamlit as st
from fastapi import FastAPI, Body, Request
import torch
import threading
import uvicorn
import time
import sys 

# Initialize FastAPI app
app = FastAPI()
BACKEND_URL = "http://localhost:8000/query"


def initialize_model():
    """Initialize the Mistral language model."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return generator, tokenizer


class GlobalObjects:
    """Class to store global objects that need to be accessed by FastAPI endpoints."""
    generator = None
    tokenizer = None
    collection = None
    db_encoder = None


@app.post("/query")
async def handle_query(request: Request):
    """Handle query requests from the frontend."""
    try:
        data = await request.json()
        query = data.get("query", "")
        
        if not query:
            return {"error": "No query provided"}
        
        # Use the global objects
        with torch.no_grad():
            query_embedding = GlobalObjects.db_encoder.encode(query)

        results = GlobalObjects.collection.query(query_embeddings=[query_embedding], n_results=3)

        context = " ".join(results["documents"][0])
        prompt = f"Given the context, please answer the question: {context}\n\n So, considering that my question is: {query}\n You can give your answer here:"
        chat_prompt = GlobalObjects.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True)

        response = GlobalObjects.generator(
            chat_prompt, 
            max_new_tokens=500, 
            pad_token_id=GlobalObjects.tokenizer.eos_token_id
        )[0]["generated_text"]
        
        # Extract only the assistant's response
        response = response.split("[/INST]")[1] if "[/INST]" in response else response

        return {"response": response, "sources": results["metadatas"][0]}
    
    except Exception as e:
        return {"error": str(e)}


def run_fastapi():
    """Run the FastAPI server in a separate thread."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


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
    parser.add_argument(
        "--use_colab_tunnel", 
        action="store_true",
        help="Use Colab's built-in tunnel to expose the application"
    )

    args = parser.parse_args()

    print(f"Retrieving data...")

    chroma_client = chromadb.PersistentClient(path=args.database_path)

    try:
        collection = chroma_client.get_collection(name=args.database_name)
        
        if len(collection.get(limit=1)["documents"]) > 0:
            print(f"Database initialized successfully!")
        else:
            print("Warning: Database appears to be empty!")
    except Exception as e:
        print(f"Error accessing database: {e}")
        return

    print("Initializing model...")
    
    # Initialize model
    generator, tokenizer = initialize_model()
    
    # Set up global objects
    GlobalObjects.generator = generator
    GlobalObjects.tokenizer = tokenizer
    GlobalObjects.collection = collection
    GlobalObjects.db_encoder = SentenceTransformer('all-MiniLM-L6-v2', 
                                                device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Give FastAPI time to start
    time.sleep(3)
    print("FastAPI backend started on http://localhost:8000")
    
    # Create and run Streamlit app
    import subprocess
    streamlit_process = subprocess.Popen(["streamlit", "run", "rag_app.py", "--server.port=8501"])
    
    print("Streamlit frontend started on http://localhost:8501")
    
    # Use Colab's tunnel if requested
    if args.use_colab_tunnel:
        try:
            from google.colab import output
            print("Opening Streamlit in a new window using Colab's tunnel...")
            output.serve_kernel_port_as_iframe(8501)
        except ImportError:
            print("Not running in Google Colab or missing output module. Tunnel not created.")
    
    try:
        # Keep the main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        streamlit_process.terminate()

if __name__ == "__main__":

    main()