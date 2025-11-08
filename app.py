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

# ----------------------------
# Load NLP Models
# ----------------------------
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

# ✅ Improved summarizer: dynamically adjusts max/min length
def summarize_text(text):
    """Adaptive summarization using BART."""
    chunks = nltk.tokenize.sent_tokenize(text)
    summaries = []
    for i in range(0, len(chunks), 5):
        input_chunk = " ".join(chunks[i:i+5])
        input_length = len(input_chunk.split())

        # Dynamically set max/min length to avoid warnings
        max_len = max(60, min(0.7 * input_length, 200))
        min_len = max(25, min(0.3 * input_length, 80))

        summary = summarizer(
            input_chunk,
            max_length=int(max_len),
            min_length=int(min_len),
            do_sample=False
        )
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
    net.write_html(output_path)  # ✅ fixed safe HTML export
    return output_path

# ✅ Improved MCQ generator
def generate_mcqs(text, tokenizer, model):
    """Generate meaningful MCQs using T5."""
    input_text = "generate questions: " + text[:800]  # use more content for better results
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs,
        max_length=80,
        num_beams=5,
        num_return_sequences=5,
        no_repeat_ngram_size=2
    )

    mcqs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # clean duplicates or short nonsense lines
    mcqs = list(set([q.strip() for q in mcqs if len(q.split()) > 4]))
    return mcqs[:5]

# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
st.title("🧩 AI Study MindMapper")
st.write(
    "Convert your **study chapters or notes** into concise **Smart Notes**, "
    "**Interactive Mindmaps**, and **Auto-Generated MCQs** — built for students and teachers."
)

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("🔍 Extracting and processing text..."):
        text = extract_text(uploaded_file)

        # --- Smart Notes ---
        st.subheader("🧠 Smart Notes (Summarized Content)")
        summary = summarize_text(text)
        st.write(summary)

        # --- Mindmap ---
        st.subheader("🗺️ Mindmap Visualization")
        keywords = extract_keywords(summary, kw_model)
        st.write("**Top Keywords:**", ", ".join(keywords))

        os.makedirs("output", exist_ok=True)
        mindmap_path = create_mindmap(keywords, "output/mindmap.html")

        with open(mindmap_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=600, scrolling=True)

        # --- MCQs ---
        st.subheader("📝 Auto-generated MCQs")
        mcqs = generate_mcqs(summary, tokenizer, mcq_model)
        if mcqs:
            for i, q in enumerate(mcqs, 1):
                st.write(f"**Q{i}.** {q}")
        else:
            st.warning("⚠️ Couldn’t generate MCQs — try uploading a longer or more informative PDF.")

    st.success("✅ Mindmap and Smart Notes generated successfully!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

