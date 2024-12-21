import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import spacy
import streamlit as st
import spacy
import os


# Try to load the SpaCy model and download it if necessary
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading the SpaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load Spacy model for lemmatization and chunking
nlp = spacy.load("en_core_web_sm")

# Load the lex_glue dataset with the ledgar split
@st.cache_data
def load_data():
    dataset = load_dataset("lex_glue", "ledgar", split="train[:500]")
    return dataset

dataset = load_data()

# Preprocess documents (lemmatization and chunking)
@st.cache_data
def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        spacy_doc = nlp(doc)
        lemmatized = " ".join([token.lemma_ for token in spacy_doc if not token.is_stop])
        chunks = [chunk.text for chunk in spacy_doc.noun_chunks]
        processed_docs.append(" ".join(chunks) + " " + lemmatized)
    return processed_docs

documents = preprocess_documents(dataset['text'])

# Load SBERT model for retrieval
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the legal documents into embeddings
@st.cache_resource
def encode_documents(documents):
    document_embeddings = sbert_model.encode(documents, convert_to_tensor=True)
    return document_embeddings

document_embeddings = encode_documents(documents)

# Use FAISS for fast retrieval
faiss_index = faiss.IndexFlatL2(document_embeddings.shape[1])
faiss_index.add(document_embeddings.cpu().numpy())

def retrieve_relevant_documents(query, top_k=3):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    _, indices = faiss_index.search(query_embedding.cpu().numpy(), top_k)
    return [documents[i] for i in indices[0]]

# Load T5 model and tokenizer for summarization
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def summarize_text(text):
    input_text = "summarize: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def legal_case_summarizer(query):
    # Step 1: Retrieve relevant documents
    relevant_documents = retrieve_relevant_documents(query, top_k=3)

    # Step 2: Summarize the top documents
    summaries = []
    for doc in relevant_documents:
        summary = summarize_text(doc)
        summaries.append(summary)

    return summaries

# Streamlit UI
st.title("Legal Case Summarizer")
st.markdown("## Enter a legal query to retrieve relevant case summaries")
query = st.text_input("Legal Query:", placeholder="Enter keywords related to a legal case")

if query:
    st.markdown("### Retrieving relevant documents...")
    summaries = legal_case_summarizer(query)

    if summaries:
        st.markdown("### Summaries of the top 3 relevant cases:")
        for i, summary in enumerate(summaries, 1):
            st.markdown(f"**Summary {i}:**")
            st.write(summary)
    else:
        st.warning("No relevant documents found. Try refining your query.")
else:
    st.info("Please enter a legal query to get started.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

