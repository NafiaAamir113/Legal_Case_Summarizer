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

# Set device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer with caching
@st.cache_resource
def load_model():
    try:
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq").to(device)
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

rag_model, rag_tokenizer = load_model()

# Function to retrieve and summarize legal cases
def retrieve_and_summarize(query):
    if not query.strip():
        return "Error: Query cannot be empty."

    if rag_model is None or rag_tokenizer is None:
        return "Error: Model or tokenizer failed to load."

    try:
        # Tokenize query
        inputs = rag_tokenizer(query, return_tensors="pt")

        # Debugging: Check tokenized inputs
        if not inputs or "input_ids" not in inputs or inputs["input_ids"] is None:
            return "Error: Tokenization failed. Check your input or model."

        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to device

        # Generate summary
        with torch.no_grad():
            generated = rag_model.generate(
                input_ids=inputs["input_ids"], 
                num_return_sequences=1, 
                max_length=200  # Prevent overly short responses
            )
            summary = rag_tokenizer.decode(generated[0], skip_special_tokens=True)
            return summary if summary else "Error: Empty summary generated."

    except Exception as e:
        return f"Error during generation: {str(e)}"

# Streamlit UI
st.title("Legal Case Summarizer")
st.write("Enter a legal query below to retrieve and summarize relevant legal cases.")

query = st.text_input("Enter Legal Query:")

if query:
    st.write(f"**Generating summary for:** {query}")
    with st.spinner("Processing..."):
        summary = retrieve_and_summarize(query)
        if "Error" in summary:
            st.error(summary)
        else:
            st.subheader("Summary:")
            st.write(summary)

        else:
            st.subheader("Summary:")
            st.write(summary)


