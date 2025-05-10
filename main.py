import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains.question_answering import load_qa_chain
import re
import logging
logging.basicConfig(level=logging.INFO)
import streamlit as st
#Load all .txt files from current folder
def load_documents_from_folder(folder_path="."):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                content = file.read()
                docs.append(content)
    return docs


raw_documents = load_documents_from_folder()

# Split documents into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = []

for doc in raw_documents:
    chunks.extend(text_splitter.split_text(doc))

# Embed and store in FAISS vector store 
print(f"\n[DEBUG] Loaded {len(raw_documents)} documents")
print(f"[DEBUG] Created {len(chunks)} chunks")

if not chunks:
    raise ValueError(" No text chunks found. Check your .txt files or content.")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embedding_model)

# Function to retrieve top K results 
def retrieve_top_k(query: str, k: int = 3):
    results = vectorstore.similarity_search(query, k=k)
    return results

#  Setup Hugging Face LLM 
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=qa_pipeline)
qa_chain = load_qa_chain(llm, chain_type="stuff")
#tool implementation
def calculator_tool(query: str) -> str:
    try:
        expr = re.sub(r"[^0-9+\-*/(). ]", "", query)  # Sanitize input
        result = eval(expr)
        return f"The result is {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"

def dictionary_tool(query: str) -> str:
    fake_dictionary = {
        "entropy": "a measure of disorder or randomness",
        "neuron": "a nerve cell that transmits signals",
        "inflera": "a knowledge assistant built using LLMs"
    }
    word = query.lower().replace("define", "").strip()
    return fake_dictionary.get(word, "Definition not found.")

# ----------------- Agent Router Logic -----------------
def agent_router(query: str):
    decision_log = []

    lower_query = query.lower()
    if any(keyword in lower_query for keyword in ["calculate", "+", "-", "*", "/", "compute"]):
        decision_log.append("Routing to Calculator Tool")
        logging.info(decision_log[-1])
        return calculator_tool(query)
    
    elif "define" in lower_query:
        decision_log.append("Routing to Dictionary Tool")
        logging.info(decision_log[-1])
        return dictionary_tool(query)
    
    else:
        decision_log.append("Routing to RAG + LLM")
        logging.info(decision_log[-1])
        top_chunks = retrieve_top_k(query)

        print("\n🔍 Top 3 matching document snippets:\n")
        for i, doc in enumerate(top_chunks, start=1):
            print(f"--- Chunk {i} ---\n{doc.page_content.strip()}\n")

        return qa_chain.run(input_documents=top_chunks, question=query)

# CLI to ask questions 
if __name__ == "__main__":
    # print("\U0001F4DA Inflera Knowledge Assistant (Agentic Workflow)")
    # print("Type your question or 'exit' to quit.")

    # while True:
    #     query = input("\nYour question: ")
    #     if query.lower().strip() == "exit":
    #         break

    #     response = agent_router(query)
    #     print("\n🤖 Answer:\n", response)

    st.set_page_config(page_title="Inflera Knowledge Assistant", layout="wide")
    st.title("📚 Inflera Knowledge Assistant")

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Thinking..."):
            result = agent_router(query)
            st.markdown("### 🤖 Answer:")
            st.write(result)

# # ----------------- Document Preview -----------------
# print(f"\u2705 Found {len(raw_documents)} documents")
# for i, doc in enumerate(raw_documents):
#     print(f"\n--- Document {i+1} Preview ---")
#     print(doc.strip()[:200])




