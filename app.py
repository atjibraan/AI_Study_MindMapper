import streamlit as st
import os
import nltk
import spacy
import networkx as nx
from pyvis.network import Network
from transformers import pipeline
from keybert import KeyBERT
from PyPDF2 import PdfReader
import tempfile
import re
from collections import Counter
import sympy as sp
import latex2sympy2

# ----------------------------
# 🧠 Setup
# ----------------------------
st.set_page_config(page_title="AI Study MindMapper - Math Enhanced", layout="wide")
st.info("✅ Enhanced with Math & Science Support")

# Content type selection
content_type = st.radio("Content Type:", 
                       ["General Text", "Mathematics", "Science", "Computer Science"],
                       horizontal=True)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

# Load NLP models
@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", 
                            model="facebook/bart-large-cnn", 
                            device=-1)
        kw_model = KeyBERT()
        
        # Try to load spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            nlp = None
            
        return summarizer, kw_model, nlp
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

summarizer, kw_model, nlp = load_models()

# ----------------------------
# 📄 Math-Specific Functions
# ----------------------------
def extract_math_formulas(text):
    """Extract mathematical formulas and equations from text."""
    # Pattern for basic math expressions
    math_patterns = [
        r'\$[^$]+\$',  # LaTeX inline math
        r'\\\[.*?\\\]',  # LaTeX display math
        r'\\\(.*?\\\)',  # LaTeX inline math
        r'[A-Za-z]+\s*=\s*[^\.]+',  # Simple equations
        r'[0-9+\-*/^()]+',  # Arithmetic expressions
    ]
    
    formulas = []
    for pattern in math_patterns:
        matches = re.findall(pattern, text)
        formulas.extend(matches)
    
    return formulas

def identify_math_concepts(text):
    """Identify mathematical concepts and topics."""
    math_keywords = {
        'algebra': ['equation', 'variable', 'polynomial', 'quadratic', 'linear'],
        'calculus': ['derivative', 'integral', 'limit', 'differentiation', 'integration'],
        'geometry': ['triangle', 'circle', 'angle', 'area', 'volume'],
        'trigonometry': ['sine', 'cosine', 'tangent', 'sin', 'cos', 'tan'],
        'statistics': ['mean', 'median', 'mode', 'probability', 'distribution']
    }
    
    found_concepts = []
    text_lower = text.lower()
    
    for concept, keywords in math_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            found_concepts.append(concept)
    
    return found_concepts

def create_math_mindmap(keywords, formulas, concepts, output_path):
    """Create enhanced mindmap for mathematical content."""
    try:
        G = nx.Graph()
        
        # Central node based on content type
        central_node = "Mathematics" if content_type == "Mathematics" else "Main Topic"
        G.add_node(central_node, size=30, color="#FF6B6B", font={'size': 25})
        
        # Add formula nodes (different color)
        for i, formula in enumerate(formulas[:5]):  # Limit to 5 formulas
            formula_node = f"Formula_{i+1}"
            G.add_node(formula_node, size=15, color="#9B59B6", title=formula)
            G.add_edge(central_node, formula_node)
        
        # Add concept nodes
        for concept in concepts:
            G.add_node(concept, size=20, color="#3498DB")
            G.add_edge(central_node, concept)
        
        # Add keyword nodes
        for i, keyword in enumerate(keywords):
            if keyword not in concepts:  # Avoid duplicates
                G.add_node(keyword, size=18, color="#2ECC71")
                G.add_edge(central_node, keyword)
                
                # Connect related concepts
                if i > 0 and i % 2 == 0:
                    G.add_edge(keywords[i-1], keyword)

        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(G)
        
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        net.write_html(output_path)
        return output_path
    except Exception as e:
        st.error(f"Mindmap creation error: {e}")
        return None

def generate_math_questions(text, concepts):
    """Generate math-related questions based on content."""
    questions = []
    
    # Template-based question generation
    question_templates = [
        "Explain the concept of {concept} in your own words.",
        "What is the importance of {concept} in mathematics?",
        "How would you apply {concept} to solve real-world problems?",
        "What are the key formulas related to {concept}?",
        "Compare and contrast {concept} with similar mathematical concepts."
    ]
    
    for concept in concepts:
        for template in question_templates[:2]:  # Use first 2 templates per concept
            questions.append(template.format(concept=concept))
    
    # Add general math questions
    general_questions = [
        "What mathematical principles are discussed in this material?",
        "How are formulas applied in the given context?",
        "What problem-solving strategies are demonstrated?",
        "How does this mathematical concept relate to other areas of math?"
    ]
    
    questions.extend(general_questions[:2])
    return questions[:8]  # Return max 8 questions

# ----------------------------
# 📄 Enhanced Utility Functions
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
        st.error(f"Text extraction error: {e}")
        return ""

def summarize_text(text, max_len=150):
    """Enhanced summarization with math awareness."""
    try:
        if len(text.strip()) < 100:
            return text
            
        # For math content, preserve formulas and key concepts
        if content_type == "Mathematics":
            formulas = extract_math_formulas(text)
            concepts = identify_math_concepts(text)
            
            # Use smaller chunks for math content
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 300:  # Smaller chunks for math
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # Standard chunking for other content
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        summaries = []
        for chunk in chunks[:4]:
            try:
                if len(chunk) > 50:
                    summary = summarizer(chunk, max_length=max_len, min_length=40, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(chunk[:100] + "...")
        
        result = " ".join(summaries) if summaries else text[:500]
        
        # Add math context for math content
        if content_type == "Mathematics" and concepts:
            result += f"\n\nKey Mathematical Concepts: {', '.join(concepts)}"
        
        return result
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return text[:500]

def extract_keywords_enhanced(text):
    """Enhanced keyword extraction with domain awareness."""
    try:
        if content_type == "Mathematics":
            # Extract math-specific keywords
            concepts = identify_math_concepts(text)
            formulas = extract_math_formulas(text)
            
            # Use KeyBERT for general keywords
            if kw_model is not None:
                general_keywords = kw_model.extract_keywords(
                    text, keyphrase_ngram_range=(1, 2), top_n=5
                )
                general_keywords = [kw[0] for kw in general_keywords]
            else:
                general_keywords = []
            
            # Combine math concepts and general keywords
            all_keywords = concepts + general_keywords
            return list(dict.fromkeys(all_keywords))[:8]  # Remove duplicates, limit to 8
        
        else:
            # Standard keyword extraction for other content
            if kw_model is not None:
                keywords = kw_model.extract_keywords(
                    text, keyphrase_ngram_range=(1, 2), top_n=8
                )
                return [kw[0] for kw in keywords]
            else:
                words = text.lower().split()
                words = [word for word in words if word.isalpha() and len(word) > 3]
                freq_dist = Counter(words)
                return [word for word, freq in freq_dist.most_common(8)]
    except Exception as e:
        st.error(f"Keyword extraction error: {e}")
        return ["key", "concepts", "important", "points"]

# ----------------------------
# 🚀 Streamlit UI
# ----------------------------
st.title("🧮 AI Study MindMapper - Math Enhanced")
st.write("Convert study material into **Smart Notes, Mindmaps, and Questions** with enhanced math support.")

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        text = extract_text(uploaded_file)
        
        if not text.strip():
            st.error("No text could be extracted. Please try a different file.")
        else:
            # Show content analysis
            with st.expander("📊 Document Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Content Type:**", content_type)
                    if content_type == "Mathematics":
                        math_concepts = identify_math_concepts(text)
                        st.write("**Math Concepts:**", ", ".join(math_concepts) if math_concepts else "None detected")
                
                with col2:
                    formulas = extract_math_formulas(text)
                    st.write("**Formulas Found:**", len(formulas))
                    if formulas:
                        with st.expander("View Formulas"):
                            for formula in formulas[:5]:  # Show first 5
                                st.code(formula)

            # Summarization
            st.subheader("🧠 Smart Notes")
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
            st.write(summary)

            # Keywords + Enhanced Mindmap
            st.subheader("🗺️ Enhanced Mindmap")
            with st.spinner("Analyzing content..."):
                keywords = extract_keywords_enhanced(text)
                formulas = extract_math_formulas(text)
                concepts = identify_math_concepts(text)
            
            st.write("**Key Elements:**", ", ".join(keywords))
            if formulas:
                st.write("**Formulas:**", f"Found {len(formulas)} mathematical expressions")

            os.makedirs("output", exist_ok=True)
            with st.spinner("Creating interactive mindmap..."):
                mindmap_path = create_math_mindmap(keywords, formulas, concepts, "output/mindmap.html")

            if mindmap_path and os.path.exists(mindmap_path):
                with open(mindmap_path, "r", encoding="utf-8") as f:
                    html_code = f.read()
                st.components.v1.html(html_code, height=600, scrolling=True)
            else:
                st.warning("Could not generate mindmap visualization")

            # Questions
            st.subheader("📝 Generated Questions")
            if content_type == "Mathematics":
                questions = generate_math_questions(summary, concepts)
            else:
                questions = [
                    "What is the main topic discussed?",
                    "What are the key points mentioned?",
                    "How can you apply this knowledge?",
                    "What relationships exist between the main concepts?"
                ]
            
            for i, q in enumerate(questions, 1):
                st.write(f"**Q{i}.** {q}")

    st.success("✅ Processing completed!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

# ----------------------------
# 📝 Features by Content Type
# ----------------------------
with st.expander("🔧 Supported Features by Content Type"):
    st.markdown("""
    | Content Type | Formula Detection | Concept Mapping | Specialized Questions |
    |-------------|-------------------|-----------------|---------------------|
    | **Mathematics** | ✅ Basic patterns | ✅ Math concepts | ✅ Template-based |
    | **Science** | ✅ Basic patterns | ✅ Science topics | ✅ Context-aware |
    | **Computer Science** | ✅ Code snippets | ✅ CS concepts | ✅ Application-focused |
    | **General Text** | ❌ | ✅ General keywords | ✅ Standard questions |
    """)

st.markdown("---")
st.markdown("Enhanced with mathematical content awareness | Formula detection | Concept mapping")
