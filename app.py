import streamlit as st
import os
import nltk
import spacy
import networkx as nx
from pyvis.network import Network
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from PyPDF2 import PdfReader

# ----------------------------
# 🧠 Setup
# ----------------------------
st.set_page_config(page_title="AI Study MindMapper", layout="wide")
device = "cpu"
st.info(f"✅ Device set to use {device}")

nltk.download('punkt', quiet=True)

# Load NLP models once
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    mcq_model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
    tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
    kw_model = KeyBERT()
    nlp = spacy.load("en_core_web_sm")
    return summarizer, mcq_model, tokenizer, kw_model, nlp

summarizer, mcq_model, tokenizer, kw_model, nlp = load_models()

# ----------------------------
# 📄 Utility Functions
# ----------------------------
def extract_text(file):
    """Extract text from uploaded PDF or TXT."""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

def summarize_text(text, max_len=130):
    """Summarize long text using BART."""
    chunks = nltk.tokenize.sent_tokenize(text)
    summaries = []
    for i in range(0, len(chunks), 5):
        input_chunk = " ".join(chunks[i:i+5])
        summary = summarizer(input_chunk, max_length=max_len, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

def extract_keywords(text, kw_model):
    """Extract key terms using KeyBERT."""
    keywords = kw_model.extract_keywords(text, top_n=10, stop_words='english')
    return [kw[0] for kw in keywords]

def create_mindmap(keywords, output_path):
    """Create and save interactive mindmap."""
    G = nx.Graph()
    for word in keywords:
        G.add_node(word)
    for i in range(len(keywords) - 1):
        G.add_edge(keywords[i], keywords[i + 1])

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.write_html(output_path)  # ✅ Fixed: safer than net.show()
    return output_path

def generate_mcqs(text, tokenizer, model):
    """Generate MCQs from summary using T5 model."""
    input_text = "generate questions: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=5)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
st.title("🧩 AI Study MindMapper")
st.write("Convert study material or chapters into **Smart Notes, Mindmaps, and MCQs** using NLP + Visualization.")

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Extracting and processing text..."):
        text = extract_text(uploaded_file)

        # Summarization
        st.subheader("🧠 Smart Notes (Summarized Content)")
        summary = summarize_text(text)
        st.write(summary)

        # Keywords + Mindmap
        st.subheader("🗺️ Mindmap Visualization")
        keywords = extract_keywords(summary, kw_model)
        st.write("**Top Keywords:**", ", ".join(keywords))

        os.makedirs("output", exist_ok=True)
        mindmap_path = create_mindmap(keywords, "output/mindmap.html")

        with open(mindmap_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=600, scrolling=True)

        # MCQs
        st.subheader("📝 Auto-generated MCQs")
        mcqs = generate_mcqs(summary, tokenizer, mcq_model)
        for i, q in enumerate(mcqs, 1):
            st.write(f"**Q{i}.** {q}")

    st.success("✅ Mindmap and Smart Notes generated successfully!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

