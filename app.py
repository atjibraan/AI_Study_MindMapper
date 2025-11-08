import streamlit as st
import os
import nltk
import spacy
import networkx as nx
from pyvis.network import Network
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from keybert import KeyBERT
from PyPDF2 import PdfReader
import tempfile

# ----------------------------
# 🧠 Setup & NLTK Download
# ----------------------------
st.set_page_config(page_title="AI Study MindMapper", layout="wide")
device = "cpu"
st.info(f"✅ Device set to use {device}")

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.warning(f"⚠️ NLTK download issue: {e}. Using alternative tokenization.")
download_nltk_data()

# Load NLP models once
@st.cache_resource
def load_models():
    try:
        # Summarization model
        summarizer = pipeline("summarization", 
                            model="facebook/bart-large-cnn", 
                            tokenizer="facebook/bart-large-cnn",
                            device=-1)
        
        # Use a simpler T5 model for question generation
        st.info("🔄 Loading question generation model...")
        try:
            mcq_model_name = "valhalla/t5-small-qa-qg-hl"
            tokenizer = T5Tokenizer.from_pretrained(mcq_model_name)
            mcq_model = T5ForConditionalGeneration.from_pretrained(mcq_model_name)
            st.success("✅ Question generation model loaded!")
        except Exception as e:
            st.warning(f"⚠️ Could not load T5 model: {e}. Using fallback mode.")
            mcq_model = None
            tokenizer = None
        
        # Keyword extraction
        kw_model = KeyBERT()
        
        # SpaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.warning("⚠️ spaCy English model not found. Using basic NLP processing.")
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
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"❌ Error extracting text: {str(e)}")
        return ""

def simple_sentence_split(text):
    """Fallback sentence splitting if NLTK fails."""
    import re
    # Basic sentence splitting by punctuation
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def summarize_text(text, max_len=130):
    """Summarize long text using BART."""
    try:
        if len(text.strip()) < 100:
            return text
            
        # Try NLTK sentence tokenization first, then fallback
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
        except Exception as e:
            st.warning("⚠️ NLTK tokenization failed, using basic sentence splitting.")
            sentences = simple_sentence_split(text)
        
        # If text is short, return as is
        if len(sentences) <= 3:
            return text
            
        # Process in chunks
        chunk_size = 500
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        summaries = []
        for i, chunk in enumerate(chunks[:4]):  # Limit to first 4 chunks to avoid timeout
            try:
                if len(chunk) > 50:  # Only summarize substantial chunks
                    summary = summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                    st.write(f"✅ Processed chunk {i+1}/{min(len(chunks), 4)}")
            except Exception as e:
                st.warning(f"⚠️ Could not summarize chunk {i+1}: {e}")
                # Use first 100 chars of chunk as fallback
                summaries.append(chunk[:100] + "...")
        
        result = " ".join(summaries) if summaries else text[:500]
        return result
    except Exception as e:
        st.error(f"❌ Error in summarization: {str(e)}")
        # Return a simple summary by taking first few sentences
        try:
            sentences = simple_sentence_split(text)
            return " ".join(sentences[:3])
        except:
            return text[:500]

def extract_keywords(text, top_n=8):
    """Extract key terms using multiple methods."""
    try:
        if not text.strip():
            return []
            
        # Use KeyBERT if available, else fallback to basic extraction
        if kw_model is not None:
            keywords = kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                diversity=0.6
            )
            return [kw[0] for kw in keywords]
        else:
            # Basic keyword extraction fallback
            try:
                words = nltk.word_tokenize(text.lower())
            except:
                # Basic word splitting if tokenization fails
                words = text.lower().split()
            words = [word for word in words if word.isalpha() and len(word) > 3]
            from collections import Counter
            freq_dist = Counter(words)
            return [word for word, freq in freq_dist.most_common(top_n)]
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
        G.add_node("Main Topic", size=25, color="#FF6B6B", font={'size': 20})
        
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
        
        # Set physics configuration for better layout
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springConstant": 0.04,
              "springLength": 95
            }
          }
        }
        """)
        
        net.write_html(output_path)
        return output_path
    except Exception as e:
        st.error(f"❌ Error creating mindmap: {str(e)}")
        return None

def generate_mcqs_fallback(text, num_questions=3):
    """Fallback MCQ generation using pattern-based approach."""
    try:
        # Use our simple sentence splitter
        sentences = simple_sentence_split(text)
        questions = []
        
        for sentence in sentences[:num_questions]:
            words = sentence.split()
            if len(words) > 5:  # Only use substantial sentences
                # Simple question generation pattern
                if "is" in sentence.lower():
                    parts = sentence.lower().split('is')
                    if len(parts) > 1 and len(parts[0].split()) < 6:
                        question = f"What is {parts[1].split('.')[0].strip()}?"
                        questions.append(question)
                elif "are" in sentence.lower():
                    parts = sentence.lower().split('are')
                    if len(parts) > 1 and len(parts[0].split()) < 6:
                        question = f"What are {parts[1].split('.')[0].strip()}?"
                        questions.append(question)
                elif len(questions) < num_questions:
                    question = f"What is the significance of: {sentence[:80]}?"
                    questions.append(question)
        
        # Ensure we have enough questions
        while len(questions) < num_questions:
            questions.append(f"Question {len(questions)+1}: What key concept did you learn from this material?")
        
        return questions[:num_questions]
    except Exception as e:
        return [
            "What is the main topic discussed?",
            "What are the key points mentioned?",
            "How does this information relate to the overall subject?"
        ]

def generate_mcqs(text, tokenizer, model, num_questions=3):
    """Generate MCQs using T5 model if available, else use fallback."""
    if model is None or tokenizer is None:
        return generate_mcqs_fallback(text, num_questions)
    
    try:
        if not text.strip() or len(text) < 50:
            return generate_mcqs_fallback(text, num_questions)
            
        # Use first 400 characters for question generation
        input_text = text[:400]
        
        # Different prompt format for this model
        prompt = f"generate question: {input_text}"
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs, 
            max_length=64, 
            num_return_sequences=num_questions,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        questions = []
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            if question and question not in questions and len(question) > 10:
                questions.append(question)
                
        return questions if questions else generate_mcqs_fallback(text, num_questions)
    except Exception as e:
        st.warning(f"⚠️ MCQ generation failed: {e}. Using fallback.")
        return generate_mcqs_fallback(text, num_questions)

# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
st.title("🧩 AI Study MindMapper")
st.write("Convert study material or chapters into **Smart Notes, Mindmaps, and MCQs** using NLP + Visualization.")

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Extracting and processing text..."):
        text = extract_text(uploaded_file)
        
        if not text.strip():
            st.error("❌ No text could be extracted from the file. Please try a different file.")
        else:
            # Show original text preview
            with st.expander("📊 Original Text Preview"):
                st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200, key="preview")

            # Summarization
            st.subheader("🧠 Smart Notes (Summarized Content)")
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
            st.write(summary)

            # Keywords + Mindmap
            st.subheader("🗺️ Mindmap Visualization")
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(summary)
            st.write("**Top Keywords:**", ", ".join(keywords))

            # Create output directory
            os.makedirs("output", exist_ok=True)
            with st.spinner("Creating mindmap..."):
                mindmap_path = create_mindmap(keywords, "output/mindmap.html")

            if mindmap_path and os.path.exists(mindmap_path):
                with open(mindmap_path, "r", encoding="utf-8") as f:
                    html_code = f.read()
                st.components.v1.html(html_code, height=600, scrolling=True)
                
                # Download button for mindmap
                with open(mindmap_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Mindmap",
                        data=f,
                        file_name="mindmap.html",
                        mime="text/html"
                    )
            else:
                st.warning("Could not generate mindmap visualization")

            # MCQs
            st.subheader("📝 Auto-generated MCQs")
            with st.spinner("Generating practice questions..."):
                mcqs = generate_mcqs(summary, tokenizer, mcq_model)
            for i, q in enumerate(mcqs, 1):
                st.write(f"**Q{i}.** {q}")

    st.success("✅ Processing completed!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

# ----------------------------
# 📝 Instructions & Status
# ----------------------------
with st.expander("ℹ️ How to use this app & System Status"):
    st.markdown("""
    ### 📖 How to use:
    1. **Upload a PDF or TXT file** containing your study material
    2. **Wait for processing** - the app will:
       - Extract and summarize the text
       - Identify key concepts and create a mindmap
       - Generate practice questions
    3. **Use the outputs**:
       - Study the summarized notes
       - Explore relationships in the interactive mindmap
       - Test your knowledge with the generated MCQs

    ### 🔧 System Status:
    """)
    
    # Model status
    status_emoji = "✅" if summarizer else "❌"
    st.write(f"{status_emoji} Summarization Model: {'Loaded' if summarizer else 'Failed'}")
    
    status_emoji = "✅" if mcq_model else "⚠️"
    st.write(f"{status_emoji} Question Generation: {'Loaded' if mcq_model else 'Using fallback mode'}")
    
    status_emoji = "✅" if kw_model else "⚠️"
    st.write(f"{status_emoji} Keyword Extraction: {'Loaded' if kw_model else 'Using basic mode'}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Transformers, and NetworkX | Robust fallback modes included")
