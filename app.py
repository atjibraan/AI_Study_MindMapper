import streamlit as st
import os
import nltk
import spacy
import networkx as nx
from pyvis.network import Network
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from PyPDF2 import PdfReader
import tempfile

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
    try:
        # Summarization model
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        
        # MCQ generation model with explicit tokenizer
        mcq_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        tokenizer = AutoTokenizer.from_pretrained(mcq_model_name, use_fast=True)
        mcq_model = AutoModelForSeq2SeqLM.from_pretrained(mcq_model_name)
        
        # Keyword extraction
        kw_model = KeyBERT()
        
        # SpaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("⚠️ spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
            nlp = None
            
        return summarizer, mcq_model, tokenizer, kw_model, nlp
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

summarizer, mcq_model, tokenizer, kw_model, nlp = load_models()

# ----------------------------
# 📄 Utility Functions
# ----------------------------
def extract_text(file):
    """Extract text from uploaded PDF or TXT."""
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"❌ Error extracting text: {str(e)}")
        return ""

def summarize_text(text, max_len=130):
    """Summarize long text using BART."""
    try:
        if len(text.strip()) < 100:
            return text
            
        chunks = nltk.tokenize.sent_tokenize(text)
        summaries = []
        
        for i in range(0, len(chunks), 3):  # Reduced chunk size for stability
            input_chunk = " ".join(chunks[i:i+3])
            if len(input_chunk) > 50:  # Ensure chunk is long enough
                summary = summarizer(input_chunk, max_length=max_len, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries) if summaries else text[:500]  # Fallback
    except Exception as e:
        st.error(f"❌ Error in summarization: {str(e)}")
        return text[:500]  # Return first 500 chars as fallback

def extract_keywords(text, kw_model, top_n=8):
    """Extract key terms using KeyBERT."""
    try:
        if not text.strip():
            return []
            
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2),  # Allow 1-2 word phrases
            stop_words='english',
            top_n=top_n,
            diversity=0.5  # Ensure diverse keywords
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        st.error(f"❌ Error extracting keywords: {str(e)}")
        return ["concepts", "learning", "study", "materials", "key", "points"]

def create_mindmap(keywords, output_path):
    """Create and save interactive mindmap."""
    try:
        if not keywords:
            st.warning("No keywords to create mindmap")
            return None
            
        G = nx.Graph()
        
        # Add central node
        G.add_node("Main Topic", size=25, color="#FF6B6B")
        
        # Add keyword nodes and connect to central node
        for i, word in enumerate(keywords):
            G.add_node(word, size=20, color="#4ECDC4")
            G.add_edge("Main Topic", word)
            
            # Connect some keywords to each other for better visualization
            if i > 0 and i % 2 == 0:
                G.add_edge(keywords[i-1], word)

        net = Network(
            height="600px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            directed=False
        )
        
        net.from_nx(G)
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        net.write_html(output_path)
        return output_path
    except Exception as e:
        st.error(f"❌ Error creating mindmap: {str(e)}")
        return None

def generate_mcqs(text, tokenizer, model, num_questions=3):
    """Generate MCQs from summary using T5 model."""
    try:
        if not text.strip() or len(text) < 50:
            return ["Sample question: What is the main topic discussed?"]
            
        # Clean and prepare text
        input_text = text[:512]  # Limit input length
        prompt = f"generate question: {input_text}"
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs, 
            max_length=64, 
            num_return_sequences=num_questions,
            num_beams=4,
            early_stopping=True
        )
        
        questions = []
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            if question and question not in questions:
                questions.append(question)
                
        return questions if questions else ["What are the key concepts in this text?"]
    except Exception as e:
        st.error(f"❌ Error generating MCQs: {str(e)}")
        return ["What is the main idea presented in the text?"]

# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
st.title("🧩 AI Study MindMapper")
st.write("Convert study material or chapters into **Smart Notes, Mindmaps, and MCQs** using NLP + Visualization.")

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    # Check if models loaded successfully
    if any(model is None for model in [summarizer, mcq_model, tokenizer, kw_model]):
        st.error("❌ Some models failed to load. Please check the error messages above.")
    else:
        with st.spinner("Extracting and processing text..."):
            text = extract_text(uploaded_file)
            
            if not text.strip():
                st.error("❌ No text could be extracted from the file. Please try a different file.")
            else:
                st.subheader("📊 Original Text Preview")
                st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)

                # Summarization
                st.subheader("🧠 Smart Notes (Summarized Content)")
                summary = summarize_text(text)
                st.write(summary)

                # Keywords + Mindmap
                st.subheader("🗺️ Mindmap Visualization")
                keywords = extract_keywords(summary, kw_model)
                st.write("**Top Keywords:**", ", ".join(keywords))

                # Create output directory
                os.makedirs("output", exist_ok=True)
                mindmap_path = create_mindmap(keywords, "output/mindmap.html")

                if mindmap_path and os.path.exists(mindmap_path):
                    with open(mindmap_path, "r", encoding="utf-8") as f:
                        html_code = f.read()
                    st.components.v1.html(html_code, height=600, scrolling=True)
                else:
                    st.warning("Could not generate mindmap visualization")

                # MCQs
                st.subheader("📝 Auto-generated MCQs")
                mcqs = generate_mcqs(summary, tokenizer, mcq_model)
                for i, q in enumerate(mcqs, 1):
                    st.write(f"**Q{i}.** {q}")

        st.success("✅ Processing completed!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

# ----------------------------
# 📝 Instructions
# ----------------------------
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Upload a PDF or TXT file** containing your study material
    2. **Wait for processing** - the app will:
       - Extract and summarize the text
       - Identify key concepts and create a mindmap
       - Generate practice questions
    3. **Use the outputs**:
       - Study the summarized notes
       - Explore relationships in the interactive mindmap
       - Test your knowledge with the generated MCQs
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Transformers, and NetworkX")
