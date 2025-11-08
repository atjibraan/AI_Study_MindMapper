import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from keybert import KeyBERT
import spacy
from spacy.cli import download
from PyPDF2 import PdfReader
import random

# ----------------------------
# ⚙️ Load Models (cached for speed)
# ----------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    mcq_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    kw_model = KeyBERT()

    # ✅ Fix: Auto-download SpaCy model if missing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    return summarizer, mcq_model, tokenizer, kw_model, nlp


summarizer, mcq_model, tokenizer, kw_model, nlp = load_models()

# ----------------------------
# 📄 Utility Functions
# ----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


def summarize_text(text, max_length=150):
    input_length = len(text.split())
    adjusted_length = min(max_length, max(60, input_length // 2))
    summary = summarizer(text, max_length=adjusted_length, min_length=40, do_sample=False)
    return summary[0]["summary_text"]


def generate_keywords(text, kw_model):
    keywords = kw_model.extract_keywords(text, top_n=10)
    return [kw[0] for kw in keywords]


def generate_mcqs(text, keywords, n=5):
    mcqs = []
    sentences = [sent.text for sent in nlp(text).sents]

    for i, kw in enumerate(keywords[:n]):
        sentence = random.choice(sentences)
        question = sentence.replace(kw, "______")
        if "______" not in question:
            continue

        distractors = random.sample(keywords, min(3, len(keywords)))
        if kw in distractors:
            distractors.remove(kw)
        options = distractors + [kw]
        random.shuffle(options)

        mcqs.append({
            "question": question,
            "options": options,
            "answer": kw
        })
    return mcqs


# ----------------------------
# 🎨 Streamlit App UI
# ----------------------------
st.set_page_config(page_title="AI Study MindMapper", layout="wide")

st.title("🧠 AI Study MindMapper")
st.markdown("Upload your study PDF — get a **smart summary**, **key topics**, and **auto-generated MCQs**!")

uploaded_file = st.file_uploader("📘 Upload your study material (PDF only)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if len(text) < 200:
        st.warning("⚠️ The uploaded PDF doesn't contain enough text for summarization.")
    else:
        st.success("✅ Text successfully extracted!")

        with st.spinner("Summarizing the content..."):
            summary = summarize_text(text)

        st.subheader("📄 Summary")
        st.write(summary)

        with st.spinner("Extracting important keywords..."):
            keywords = generate_keywords(summary, kw_model)

        st.subheader("🔑 Key Topics")
        st.write(", ".join(keywords))

        with st.spinner("Generating multiple choice questions..."):
            mcqs = generate_mcqs(summary, keywords)

        st.subheader("❓ Auto-Generated MCQs")
        for i, mcq in enumerate(mcqs, 1):
            st.markdown(f"**Q{i}.** {mcq['question']}")
            for opt in mcq['options']:
                st.markdown(f"- {opt}")
            st.markdown(f"**Answer:** ✅ {mcq['answer']}")
            st.markdown("---")

else:
    st.info("👆 Please upload a PDF file to get started.")
