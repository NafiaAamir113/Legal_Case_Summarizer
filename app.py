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
from transformers import BartForConditionalGeneration, BartTokenizer

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

# Function to generate summaries
def summarize_text(text):
    if not text.strip():
        return "Error: Query cannot be empty."

    if model is None or tokenizer is None:
        return "Error: Model or tokenizer failed to load."

    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to device

        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                max_length=250,
                num_return_sequences=1,
                length_penalty=2.0,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary if summary else "Error: Empty summary generated."
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Streamlit UI
st.title("Legal Case Summarizer")
st.write("Enter a legal query below to retrieve and summarize relevant legal cases.")

query = st.text_area("Enter Legal Query:")

if query:
    st.write(f"**Generating summary for:** {query}")
    with st.spinner("Processing..."):
        summary = summarize_text(query)
        if "Error" in summary:
            st.error(summary)
        else:
            st.subheader("Summary:")
            st.write(summary)



