
# 🧠 AI Study MindMapper
# Smart Notes Summarizer + Mindmap Generator + MCQ Creator


import streamlit as st
import PyPDF2
from transformers import pipeline
from keybert import KeyBERT
import networkx as nx
from pyvis.network import Network
import nltk
from nltk.corpus import wordnet
import random
import os

# -----------------------------
# 📦 Setup
# -----------------------------
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="AI Study MindMapper", layout="wide")
st.title("🧠 AI Study MindMapper")
st.write("Summarize study notes, visualize mindmaps, and generate MCQs using AI.")

# -----------------------------
# 📂 Helper Functions
# -----------------------------
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text


@st.cache_resource
def load_summarizer():
    """Load the summarization model"""
    return pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text, summarizer, max_len=300):
    """Summarize long text into concise notes"""
    if len(text.split()) < 50:
        return text
    try:
        summary = summarizer(text, max_length=max_len, min_length=80, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"


@st.cache_resource
def load_keybert():
    """Load KeyBERT model"""
    return KeyBERT()


def extract_keywords(text, kw_model, num_keywords=10):
    """Extract top keyphrases from text"""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')
    return [kw[0] for kw in keywords]


def create_mindmap(keywords, output_path="mindmap.html"):
    """Generate and save an interactive mindmap"""
    G = nx.Graph()
    G.add_node("Main Topic")

    for kw in keywords:
        G.add_node(kw)
        G.add_edge("Main Topic", kw)

    net = Network(notebook=False, height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    net.show(output_path)
    return output_path


def generate_mcqs(keywords):
    """Generate basic MCQs from keywords"""
    mcqs = []
    for kw in keywords:
        syns = wordnet.synsets(kw)
        options = [kw]
        if syns:
            for s in syns[:3]:
                lemma = s.lemmas()[0].name().replace('_', ' ')
                if lemma.lower() != kw.lower():
                    options.append(lemma)
        while len(options) < 4:
            options.append(f"Option{random.randint(1,100)}")
        random.shuffle(options)
        mcqs.append({
            "question": f"What is related to **{kw}**?",
            "options": options,
            "answer": kw
        })
    return mcqs


# -----------------------------
# 🧠 Main App Logic
# -----------------------------
uploaded_file = st.file_uploader("📘 Upload your study PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    if len(text.strip()) == 0:
        st.error("No readable text found in the PDF. Try another file.")
    else:
        st.success("✅ Text extracted successfully!")

        # Summarization
        st.subheader("📄 Summarized Notes")
        summarizer = load_summarizer()
        summary = summarize_text(text, summarizer)
        st.write(summary)

        # Keywords
        st.subheader("🔑 Key Topics")
        kw_model = load_keybert()
        keywords = extract_keywords(summary, kw_model)
        st.write(", ".join(keywords))

        # Mindmap
        st.subheader("🧩 Mindmap Visualization")
        os.makedirs("output", exist_ok=True)
        mindmap_path = create_mindmap(keywords, "output/mindmap.html")
        st.components.v1.html(open(mindmap_path, 'r', encoding='utf-8').read(), height=600)

        # MCQs
        st.subheader("📝 Auto-Generated MCQs")
        mcqs = generate_mcqs(keywords)
        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            for opt in q['options']:
                st.markdown(f"- {opt}")
            st.markdown(f"**Answer:** {q['answer']}")
            st.markdown("---")

        # Download Options
        st.download_button("⬇️ Download Summary", summary, file_name="summary.txt")
else:
    st.info("👆 Upload a PDF to get started.")
