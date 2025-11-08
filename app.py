import streamlit as st
import os
import nltk
import networkx as nx
from pyvis.network import Network
from transformers import pipeline, Conversation
from keybert import KeyBERT
from PyPDF2 import PdfReader
import re
from collections import Counter
import datetime

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
        
        # Load conversational model for interactive Q&A
        conversational_pipeline = pipeline(
            "conversational",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium"
        )
        
        return summarizer, kw_model, conversational_pipeline
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

summarizer, kw_model, conversational_pipeline = load_models()

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
# 🆕 INTERACTIVE CONVERSATIONAL Q&A SYSTEM
# ----------------------------
def initialize_conversation():
    """Initialize or reset the conversation history."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def add_message(role, message):
    """Add a message to the conversation history."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        'role': role,
        'message': message,
        'timestamp': timestamp
    })

def get_conversational_response(user_input, document_context=""):
    """Get a conversational response using the dialogue model."""
    try:
        # Create conversation context
        if document_context:
            context = f"Based on this study material: {document_context[:1000]}. "
        else:
            context = ""
        
        # Add context to user input
        enhanced_input = context + user_input
        
        # Create conversation object
        conversation = Conversation(enhanced_input)
        
        # Get response from conversational model
        result = conversational_pipeline(
            conversation,
            max_length=1000,
            pad_token_id=conversational_pipeline.tokenizer.eos_token_id
        )
        
        return result.generated_responses[-1]
    
    except Exception as e:
        return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

def get_smart_response(user_input, document_text, content_type):
    """Smart response system that combines multiple approaches."""
    user_input_lower = user_input.lower().strip()
    
    # Knowledge base for common questions
    knowledge_base = {
        # Why questions
        "why should i study algebra": "Algebra is fundamental because it teaches you how to think logically and solve problems systematically. It's the foundation for advanced mathematics, computer science, engineering, and even everyday tasks like budgeting and planning. Algebra helps develop critical thinking skills that are valuable in any career!",
        "why is math important": "Mathematics is crucial because it develops problem-solving skills, logical thinking, and analytical abilities. It's used in virtually every field - from science and engineering to finance and data analysis. Math helps us understand patterns, make predictions, and solve real-world problems efficiently.",
        "why study calculus": "Calculus is essential for understanding change and motion. It's used in physics, engineering, economics, computer graphics, and even medicine. Calculus helps model real-world phenomena like population growth, object motion, and optimization problems.",
        
        # What questions
        "what is a quadratic equation": "A quadratic equation is a second-degree polynomial equation of the form ax² + bx + c = 0. It's called 'quadratic' because 'quadratus' means square in Latin, and the variable gets squared (x²). These equations often appear in physics, engineering, and optimization problems!",
        "what is calculus": "Calculus is the mathematics of change. It has two main branches: Differential Calculus (studying rates of change) and Integral Calculus (studying accumulation). Think of it as the math that describes how things move, grow, and change over time!",
        
        # How questions
        "how to study math effectively": "Great question! Here are some tips: 1) Practice regularly with different problems, 2) Understand concepts rather than memorizing, 3) Connect math to real-world applications, 4) Don't be afraid to make mistakes - they're learning opportunities, 5) Break complex problems into smaller steps!",
        "how does this relate to real life": "Mathematics is everywhere! Algebra helps with budgeting and planning, geometry with design and construction, statistics with data analysis, and calculus with understanding change in systems like population growth or object motion!",
    }
    
    # Check knowledge base first
    for question, answer in knowledge_base.items():
        if question in user_input_lower:
            return answer
    
    # Check for question patterns and provide helpful responses
    if any(word in user_input_lower for word in ['why', 'importance', 'important', 'purpose']):
        concepts = identify_math_concepts(document_text)
        if concepts:
            return f"That's a great question! Based on your document, we're discussing {', '.join(concepts)}. These concepts are important because they form the foundation for more advanced topics and have practical applications in many fields. Would you like me to explain the importance of any specific concept in more detail?"
        else:
            return "That's an excellent question about the importance of this topic! Understanding why we study something helps with motivation and deeper learning. Could you tell me which specific concept you'd like to understand better?"
    
    elif any(word in user_input_lower for word in ['how', 'method', 'approach', 'technique']):
        return "I'd be happy to help you understand the methods and approaches! Could you specify which particular technique or process you'd like me to explain? For example, are you asking about problem-solving strategies, calculation methods, or conceptual approaches?"
    
    elif any(word in user_input_lower for word in ['what', 'define', 'definition']):
        concepts = identify_math_concepts(document_text)
        if concepts:
            return f"I can help define those concepts! Your document mentions {', '.join(concepts)}. Which specific term would you like me to explain in detail?"
        else:
            return "I'd be happy to define mathematical concepts for you! Could you specify which term or concept you'd like me to explain?"
    
    # Use conversational model as fallback
    return get_conversational_response(user_input, document_text)

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
st.title("🧮 AI Study MindMapper - Interactive Edition")
st.write("Convert study material into **Smart Notes, Mindmaps, and have interactive conversations**!")

uploaded_file = st.file_uploader("📁 Upload PDF or Text File", type=["pdf", "txt"])

# Initialize conversation
initialize_conversation()

if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        text = extract_text(uploaded_file)
        st.session_state.document_text = text  # Store text for Q&A
        
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

            # Auto-generated Questions
            st.subheader("📝 Generated Questions")
            questions = generate_domain_questions(summary, concepts, math_elements)
            
            for i, q in enumerate(questions, 1):
                st.write(f"**Q{i}.** {q}")

            # 🆕 INTERACTIVE CONVERSATIONAL Q&A
            st.subheader("💬 Interactive Study Assistant")
            st.write("Chat with me about your study material! Ask questions, seek explanations, or discuss concepts.")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
                    if chat['role'] == 'user':
                        st.markdown(f"**You** ({chat['timestamp']}): {chat['message']}")
                    else:
                        st.markdown(f"**Assistant** ({chat['timestamp']}): {chat['message']}")
            
            # Chat input
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input(
                    "Type your message:",
                    placeholder="Ask me anything about the material...",
                    key="chat_input"
                )
            with col2:
                send_button = st.button("Send", use_container_width=True)
            
            if send_button and user_input:
                # Add user message to history
                add_message('user', user_input)
                
                # Get AI response
                with st.spinner("Thinking..."):
                    response = get_smart_response(user_input, text, content_type)
                    add_message('assistant', response)
                
                # Rerun to update chat display
                st.rerun()
            
            # Quick suggestion buttons
            st.write("**Quick questions you can ask:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Why study this?", use_container_width=True):
                    add_message('user', "Why should I study this material?")
                    response = get_smart_response("Why should I study this material?", text, content_type)
                    add_message('assistant', response)
                    st.rerun()
            with col2:
                if st.button("Key concepts?", use_container_width=True):
                    add_message('user', "What are the key concepts?")
                    response = get_smart_response("What are the key concepts?", text, content_type)
                    add_message('assistant', response)
                    st.rerun()
            with col3:
                if st.button("Study tips?", use_container_width=True):
                    add_message('user', "How should I study this effectively?")
                    response = get_smart_response("How should I study this effectively?", text, content_type)
                    add_message('assistant', response)
                    st.rerun()

    st.success("✅ Processing completed!")

else:
    st.info("👆 Upload a PDF or TXT file to get started!")

# ----------------------------
# 📝 Features
# ----------------------------
with st.expander("🔧 Features"):
    st.markdown("""
    ### 🆕 Interactive Study Assistant
    - **Conversational Q&A** - Chat naturally like we're doing now!
    - **Remembers context** - Follows up on previous questions
    - **Smart responses** - Answers "why", "how", and "what" questions
    - **Quick suggestions** - Pre-built questions to get you started
    - **Real-time chat** - See the conversation history
    
    **Now you can ask questions like:**
    - "Why should I study algebra?"
    - "How does this relate to real life?"
    - "What's the most important concept here?"
    - "Can you explain this in simpler terms?"
    - "How should I study this effectively?"
    """)

st.markdown("---")
st.markdown("Interactive Study Assistant | Conversational Q&A | Smart Learning Companion")
