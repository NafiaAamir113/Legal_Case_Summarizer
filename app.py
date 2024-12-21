import streamlit as st
from transformers import RagTokenizer, RagSequenceForGeneration

# Initialize the RAG model
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Function to retrieve and summarize legal case based on input query
def retrieve_and_summarize(query):
    # Tokenize input query
    inputs = rag_tokenizer(query, return_tensors="pt")
    
    # Generate a summary using the RAG model
    generated = rag_model.generate(input_ids=inputs["input_ids"], num_return_sequences=1)
    
    # Decode and return the summary
    summary = rag_tokenizer.decode(generated[0], skip_special_tokens=True)
    return summary

# Streamlit UI components
st.title("Legal Case Summarizer")
st.write("Enter a legal query below to retrieve and summarize relevant legal cases.")

# User input for query
query = st.text_input("Enter Legal Query:")

# Display results when query is provided
if query:
    st.write("Generating summary for the query: ", query)
    summary = retrieve_and_summarize(query)
    st.write("### Summary:")
    st.write(summary)
