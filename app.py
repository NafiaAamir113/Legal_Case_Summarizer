# import streamlit as st
# from transformers import RagTokenizer, RagSequenceForGeneration

# # Initialize the RAG model
# rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
# rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# # Function to retrieve and summarize legal case based on input query
# def retrieve_and_summarize(query):
#     # Tokenize input query
#     inputs = rag_tokenizer(query, return_tensors="pt")
    
#     # Generate a summary using the RAG model
#     generated = rag_model.generate(input_ids=inputs["input_ids"], num_return_sequences=1)
    
#     # Decode and return the summary
#     summary = rag_tokenizer.decode(generated[0], skip_special_tokens=True)
#     return summary

# # Streamlit UI components
# st.title("Legal Case Summarizer")
# st.write("Enter a legal query below to retrieve and summarize relevant legal cases.")

# # User input for query
# query = st.text_input("Enter Legal Query:")

# # Display results when query is provided
# if query:
#     st.write("Generating summary for the query: ", query)
#     summary = retrieve_and_summarize(query)
#     st.write("### Summary:")
#     st.write(summary)


import streamlit as st
import torch
from transformers import RagTokenizer, RagSequenceForGeneration

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the RAG model and tokenizer
@st.cache_resource
def load_model():
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq").to(device)
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    return model, tokenizer

rag_model, rag_tokenizer = load_model()

# Function to retrieve and summarize legal cases based on input query
def retrieve_and_summarize(query):
    inputs = rag_tokenizer(query, return_tensors="pt").to(device)

    # Retrieve documents explicitly
    with torch.no_grad():
        docs_dict = rag_model.retriever(inputs["input_ids"], return_tensors="pt")
        context_input_ids = docs_dict["context_input_ids"].to(device)

        # Generate a response
        generated = rag_model.generate(input_ids=inputs["input_ids"], context_input_ids=context_input_ids, num_return_sequences=1)

    summary = rag_tokenizer.decode(generated[0], skip_special_tokens=True)
    return summary

# Streamlit UI components
st.title("Legal Case Summarizer")
st.write("Enter a legal query below to retrieve and summarize relevant legal cases.")

# User input for query
query = st.text_input("Enter Legal Query:")

# Display results when query is provided
if query:
    st.write(f"**Generating summary for:** {query}")
    with st.spinner("Processing..."):
        try:
            summary = retrieve_and_summarize(query)
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error("An error occurred while generating the summary.")
            st.text(str(e))

