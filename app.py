import streamlit as st
import os
import nltk
import networkx as nx
from pyvis.network import Network
from transformers import pipeline
from keybert import KeyBERT
from PyPDF2 import PdfReader
import re
from collections import Counter

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
    except:
        st.warning("NLTK punkt download failed - using fallback tokenization")

download_nltk_data()

# Load NLP models
@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", 
                            model="facebook/bart-large-cnn",
                            min_length=30,
                            max_length=150)
        kw_model = KeyBERT()
        return summarizer, kw_model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

summarizer, kw_model = load_models()

# ----------------------------
# 📄 Math-Specific Functions
# ----------------------------
def extract_math_elements(text):
    """Extract mathematical elements from text."""
    math_elements = {
        'equations': [],
        'expressions': [],
        'math_keywords': [],
        'numbers': []
    }
    
    # Patterns for math content
    equation_pattern = r'[A-Za-z]+\s*=\s*[^\.\n]+'  # Simple equations like "x = y + z"
    expression_pattern = r'[0-9+\-*/^()xyz]+'  # Math expressions
    number_pattern = r'\b\d+\.?\d*\b'  # Numbers
    
    # Math vocabulary
    math_vocab = [
        'equation', 'formula', 'theorem', 'proof', 'solve', 'calculate', 'compute',
        'algebra', 'calculus', 'geometry', 'trigonometry', 'statistics', 'probability',
        'derivative', 'integral', 'limit', 'function', 'variable', 'constant',
        'matrix', 'vector', 'graph', 'plot', 'angle', 'area', 'volume'
    ]
    
    # Extract equations
    equations = re.findall(equation_pattern, text)
    math_elements['equations'] = [eq.strip() for eq in equations[:5]]
    
    # Extract math expressions
    expressions = re.findall(expression_pattern, text)
    # Filter to only keep meaningful expressions (not just single numbers)
    meaningful_expr = [expr for expr in expressions if len(expr) > 2 and any(op in expr for op in ['+', '-', '*', '/', '^'])]
    math_elements['expressions'] = meaningful_expr[:5]
    
    # Extract numbers
    numbers = re.findall(number_pattern, text)
    math_elements['numbers'] = list(set(numbers))[:10]  # Unique numbers only
    
    # Identify math keywords
    text_lower = text.lower()
    found_keywords = [word for word in math_vocab if word in text_lower]
    math_elements['math_keywords'] = found_keywords
    
    return math_elements

def identify_math_concepts(text):
    """Identify mathematical concepts and topics."""
    math_categories = {
        'Algebra': ['equation', 'variable', 'polynomial', 'quadratic', 'linear', 'solve', 'factor'],
        'Calculus': ['derivative', 'integral', 'limit', 'differentiation', 'integration', 'function'],
        'Geometry': ['triangle', 'circle', 'angle', 'area', 'volume', 'shape', 'perimeter'],
        'Trigonometry': ['sine', 'cosine', 'tangent', 'sin', 'cos', 'tan', 'trigonometry'],
        'Statistics': ['mean', 'median', 'mode', 'probability', 'distribution', 'average', 'statistics']
    }
    
    found_concepts = []
    text_lower = text.lower()
    
    for category, keywords in math_categories.items():
        if any(keyword in text_lower for keyword in keywords):
            found_concepts.append(category)
    
    return found_concepts

def create_enhanced_mindmap(keywords, math_elements, concepts, output_path):
    """Create enhanced mindmap for mathematical content."""
    try:
        G = nx.Graph()
        
        # Central node based on content type
        central_node = "Mathematics" if content_type == "Mathematics" else "Main Topic"
        G.add_node(central_node, size=30, color="#FF6B6B", font={'size': 25})
        
        # Add equation nodes (different color)
        for i, equation in enumerate(math_elements['equations'][:3]):
            eq_node = f"Eq_{i+1}"
            G.add_node(eq_node, size=15, color="#9B59B6", title=equation, shape='box')
            G.add_edge(central_node, eq_node)
        
        # Add concept nodes
        for concept in concepts:
            G.add_node(concept, size=20, color="#3498DB", shape='diamond')
            G.add_edge(central_node, concept)
        
        # Add keyword nodes
        for i, keyword in enumerate(keywords):
            if keyword not in concepts and keyword not in math_elements['math_keywords']:
                G.add_node(keyword, size=18, color="#2ECC71")
                G.add_edge(central_node, keyword)
        
        # Add math keyword nodes
        for math_keyword in math_elements['math_keywords'][:5]:
            G.add_node(math_keyword, size=16, color="#E74C3C", shape='triangle')
            G.add_edge(central_node, math_keyword)

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

def generate_domain_questions(text, concepts, math_elements):
    """Generate domain-specific questions based on content."""
    questions = []
    
    if content_type == "Mathematics":
        # Math-specific questions
        for concept in concepts:
            questions.extend([
                f"Explain the concept of {concept.lower()} in your own words.",
                f"What are the key principles of {concept.lower()}?",
                f"How is {concept.lower()} applied in problem-solving?"
            ])
        
        # Equation-based questions
        for equation in math_elements['equations'][:2]:
            questions.append(f"What does this equation represent: '{equation}'?")
            questions.append(f"What variables are involved in: '{equation}'?")
    
    elif content_type == "Science":
        # Science questions
        science_questions = [
            "What scientific principles are discussed?",
            "How are experimental methods described?",
            "What hypotheses or theories are presented?",
            "How does this relate to real-world scientific applications?"
        ]
        questions.extend(science_questions)
    
    elif content_type == "Computer Science":
        # CS questions
        cs_questions = [
            "What programming concepts or algorithms are mentioned?",
            "How are computational problems approached?",
            "What data structures or architectures are discussed?",
            "How could these concepts be implemented in code?"
        ]
        questions.extend(cs_questions)
    
    else:
        # General questions
        general_questions = [
            "What is the main topic discussed?",
            "What are the key points mentioned?",
            "How can you apply this knowledge?",
            "What relationships exist between the main concepts?"
        ]
        questions.extend(general_questions)
    
    return questions[:6]  # Return max 6 questions

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

def simple_sentence_split(text):
    """Fallback sentence splitting if NLTK fails."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def summarize_text(text):
    """Enhanced summarization with domain awareness."""
    try:
        if len(text.strip()) < 100:
            return text
            
        # Try NLTK sentence tokenization first, then fallback
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
        except:
            sentences = simple_sentence_split(text)
        
        if len(sentences) <= 3:
            return text
            
        # Process in chunks
        chunk_size = 400
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
        for chunk in chunks[:3]:
            try:
                if len(chunk) > 50:
                    # Adjust max_length based on input length
                    input_length = len(chunk.split())
                    max_len = min(130, max(40, input_length // 2))
                    
                    summary = summarizer(
                        chunk, 
                        max_length=max_len, 
                        min_length=30, 
                        do_sample=False
                    )
                    summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(chunk[:100] + "...")
        
        result = " ".join(summaries) if summaries else text[:500]
        
        # Add domain context
        if content_type == "Mathematics":
            concepts = identify_math_concepts(text)
            if concepts:
                result += f"\n\n**Mathematical Concepts:** {', '.join(concepts)}"
        
        return result
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return text[:500]

def extract_keywords_enhanced(text):
    """Enhanced keyword extraction with domain awareness."""
    try:
        if content_type == "Mathematics":
            # Extract math-specific elements
            math_elements = extract_math_elements(text)
            concepts = identify_math_concepts(text)
            
            # Use KeyBERT for general keywords
            if kw_model is not None:
                general_keywords = kw_model.extract_keywords(
                    text, keyphrase_ngram_range=(1, 2), top_n=5
                )
                general_keywords = [kw[0] for kw in general_keywords]
            else:
                general_keywords = []
            
            # Combine all elements
            all_keywords = concepts + math_elements['math_keywords'] + general_keywords
            return list(dict.fromkeys(all_keywords))[:8]  # Remove duplicates
        
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
                    math_elements = extract_math_elements(text)
                    st.write("**Equations Found:**", len(math_elements['equations']))
                    st.write("**Math Keywords:**", len(math_elements['math_keywords']))

            # Summarization
            st.subheader("🧠 Smart Notes")
            with st.spinner("Generating summary..."):
                summary = summarize_text(text)
            st.write(summary)

            # Keywords + Enhanced Mindmap
            st.subheader("🗺️ Enhanced Mindmap")
            with st.spinner("Analyzing content..."):
                keywords = extract_keywords_enhanced(text)
                math_elements = extract_math_elements(text)
                concepts = identify_math_concepts(text)
            
            st.write("**Key Elements:**", ", ".join(keywords))
            if math_elements['equations']:
                st.write("**Equations:**", f"Found {len(math_elements['equations'])} mathematical equations")

            os.makedirs("output", exist_ok=True)
            with st.spinner("Creating interactive mindmap..."):
                mindmap_path = create_enhanced_mindmap(keywords, math_elements, concepts, "output/mindmap.html")

            if mindmap_path and os.path.exists(mindmap_path):
                with open(mindmap_path, "r", encoding="utf-8") as f:
                    html_code = f.read()
                st.components.v1.html(html_code, height=600, scrolling=True)
                
                # Download button
                with open(mindmap_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Mindmap",
                        data=f,
                        file_name="knowledge_mindmap.html",
                        mime="text/html"
                    )
            else:
                st.warning("Could not generate mindmap visualization")

            # Questions
            st.subheader("📝 Generated Questions")
            questions = generate_domain_questions(summary, concepts, math_elements)
            
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
    | Content Type | Equation Detection | Concept Mapping | Specialized Questions |
    |-------------|-------------------|-----------------|---------------------|
    | **Mathematics** | ✅ Basic equations | ✅ Math concepts | ✅ Domain-specific |
    | **Science** | ✅ Basic patterns | ✅ Science topics | ✅ Context-aware |
    | **Computer Science** | ❌ | ✅ CS concepts | ✅ Application-focused |
    | **General Text** | ❌ | ✅ General keywords | ✅ Standard questions |
    """)

st.markdown("---")
st.markdown("Enhanced with mathematical content awareness | Equation detection | Concept mapping")
