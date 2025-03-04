import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
import yaml
import base64
import requests
from io import BytesIO
from PIL import Image
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
import uuid
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import pyttsx3
import threading
import pandas as pd
import html

# Configure Streamlit page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Elle 💙 - Your Mental Health Companion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix the experimental_rerun deprecation
def rerun():
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = "AIzaSyCcWxgiWkDyBnBT-rwoMTT1O_UOeV_kaVw"
    
genai.configure(api_key=api_key)

# Set up custom CSS for nicer UI
def load_css():
    css = """
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    
    /* Therapeutic theme colors */
    .therapy-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .therapy-badge.empathy {background-color: #9fd8cb; color: #333;}
    .therapy-badge.suicide {background-color: #ff6b6b; color: white;}
    .therapy-badge.anger {background-color: #ff9e7d; color: #333;}
    .therapy-badge.motivation {background-color: #ffde7d; color: #333;}
    .therapy-badge.dbt {background-color: #a3d9ff; color: #333;}
    .therapy-badge.cbt {background-color: #d8b9ff; color: #333;}
    
    /* App container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Title styling */
    h1 {
        color: #6c5ce7;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        background-color: #f1f3f6;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid #e0e0e0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Chat message with avatar - FIXED HTML ESCAPING
def display_message(role, content, therapy_type=None):
    if role == "user":
        avatar_img = "https://avataaars.io/?avatarStyle=Circle&topType=ShortHairShortFlat&accessoriesType=Blank&hairColor=BrownDark&facialHairType=Blank&clotheType=Hoodie&clotheColor=Blue&eyeType=Default&eyebrowType=Default&mouthType=Default&skinColor=Light"
    else:
        avatar_img = "https://avataaars.io/?avatarStyle=Circle&topType=LongHairStraight&accessoriesType=Round&hairColor=BlondeGolden&facialHairType=Blank&clotheType=BlazerShirt&eyeType=Happy&eyebrowType=Default&mouthType=Smile&skinColor=Light"
    
    therapy_badge = ""
    if therapy_type:
        badge_class = "empathy"
        if "Suicide" in therapy_type:
            badge_class = "suicide"
        elif "Anger" in therapy_type:
            badge_class = "anger"
        elif "Motivation" in therapy_type:
            badge_class = "motivation"
        elif "Dialectical" in therapy_type:
            badge_class = "dbt"
        elif "Cognitive" in therapy_type:
            badge_class = "cbt"
        
        therapy_badge = f'<div class="therapy-badge {badge_class}">{therapy_type}</div>'
    
    # For multimodal content (text + images)
    if isinstance(content, dict):
        if "text" in content:
            # Use proper HTML escaping for the text
            content_html = html.escape(content["text"])
            if "image_url" in content:
                content_html += f'<img src="{content["image_url"]}" style="max-width: 100%; margin-top: 1rem; border-radius: 0.5rem;">'
        else:
            content_html = ""
    else:
        # Properly escape HTML in string content
        content_html = html.escape(str(content))
    
    message_html = f"""
    <div class="chat-message {role}">
        <div class="avatar">
            <img src="{avatar_img}">
            {therapy_badge if role == "assistant" else "User"}
        </div>
        <div class="message">{content_html}</div>
    </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)

# Save conversation summary to CSV
def save_conversation_summary(user_id, summary, therapy_types=None):
    """Save conversation summary to a CSV file"""
    try:
        # Create summaries directory if it doesn't exist
        summaries_dir = "conversation_summaries"
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
            
        # CSV file path - changed to responses.csv
        csv_path = "responses.csv"
        
        # Current timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get most used therapy type if available
        most_used_therapy = "General Support"
        if therapy_types and len(therapy_types) > 0:
            therapy_counts = {}
            for therapy in therapy_types:
                if therapy in therapy_counts:
                    therapy_counts[therapy] += 1
                else:
                    therapy_counts[therapy] = 1
            most_used_therapy = max(therapy_counts.items(), key=lambda x: x[1])[0]
        
        # Create dataframe for this summary
        new_data = pd.DataFrame({
            'user_id': [user_id[:8]],  # First 8 chars of UUID
            'timestamp': [timestamp],
            'therapy_type': [most_used_therapy],
            'summary': [summary]
        })
        
        # Append to existing or create new
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            updated_df = new_data
            
        # Save CSV directly to the root directory
        updated_df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving conversation summary: {str(e)}")
        return False

# Load prompts from YAML
def load_prompts(file_path='H:\Project Session\Sarvam AI Task\Project Sukoon\prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

try:
    prompts = load_prompts()
    prompts_loaded = True
except Exception as e:
    st.error(f"Error loading prompts: {e}")
    prompts = {
        "empathetic_agent_prompt": "You are an empathetic mental health counselor. Listen attentively and provide supportive guidance.",
        "suicide_prevention_agent_prompt": "You are a suicide prevention counselor. Provide immediate support, validation, and resources for the person in crisis.",
        "anger_prevention_agent_prompt": "You are an anger management specialist. Help the user process their emotions constructively.",
        "motivational_agent_prompt": "You are a motivational coach specializing in depression. Provide empathetic support and practical coping strategies.",
        "dbt_agent_prompt": "You are a therapist specializing in Dialectical Behavior Therapy for emotional regulation.",
        "cbt_agent_prompt": "You are a therapist specializing in Cognitive Behavioral Therapy for changing negative thought patterns.",
        "planner_agent_prompt": "You are a mental health assistant that determines which therapeutic approach is most appropriate based on user input."
    }
    prompts_loaded = False

# Database simulation for multiple users
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
    
def get_user_id():
    """Get or create a user ID for session management"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id
    
def get_user_messages(user_id):
    """Get messages for a specific user"""
    if user_id not in st.session_state.user_data:
        st.session_state.user_data[user_id] = {
            "messages": [],
            "conversation_summary": "",
            "chunks": None,
            "chunk_embeddings": None,
            "therapy_types": []  # Track therapy types used
        }
    return st.session_state.user_data[user_id]

# Multi-modal content generation
def generate_mental_health_image(prompt):
    """Generate an appropriate mental health image based on the therapy type"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        img_prompt = f"Generate a calm, supportive, non-triggering image description for: {prompt}"
        response = model.generate_content(img_prompt)
        img_description = response.text
        
        # Use an image generation API or placeholder images
        # For demonstration, let's use Unsplash API for photos based on the description
        unsplash_key = os.getenv("UNSPLASH_API_KEY", "demo")
        query = "+".join(img_description.split()[:5])  # First 5 words
        img_url = f"https://source.unsplash.com/featured/?{query}&calm,peaceful,wellness"
        
        return img_url
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        # Return a default calming image
        return "https://images.unsplash.com/photo-1579546929518-9e396f3cc809?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&w=600&q=80"

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

def get_embedding(text):
    """Get embedding for a given text using Gemini API."""
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(embedding["embedding"])

def retrieve_relevant_chunks(question, chunks, chunk_embeddings, top_n=5):
    """Find the most relevant chunks from the PDF based on similarity."""
    question_embedding = get_embedding(question)
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

def split_text_into_chunks(text):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    return splitter.split_text(text)

# Define summarization function
def generate_conversation_summary(conversation_history):
    """Summarize the conversation history to provide context."""
    if len(conversation_history) <= 2:  # No need to summarize very short conversations
        return ""
    
    # Extract just the content from messages
    conversation_text = ""
    for msg in conversation_history:
        content = msg["content"]
        if isinstance(content, dict) and "text" in content:
            conversation_text += f"{msg['role']}: {content['text']}\n"
        else:
            conversation_text += f"{msg['role']}: {content}\n"
    
    # Generate summary using Gemini
    model = genai.GenerativeModel("gemini-1.5-pro")
    summarization_prompt = f"""
    Summarize the following conversation between a user and an AI mental health assistant.
    Focus on key emotional themes, concerns, and progress made. Keep it concise.
    
    CONVERSATION:
    {conversation_text}
    
    SUMMARY:
    """
    
    try:
        response = model.generate_content(summarization_prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return None
        except sr.RequestError:
            st.error("Could not request results from speech recognition service")
            return None
        except Exception as e:
            st.error(f"Speech recognition error: {str(e)}")
            return None

# Text to speech
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")

# Main application
def main():
    # Apply CSS
    load_css()
    
    # Get user data
    user_id = get_user_id()
    user_data = get_user_messages(user_id)
    
    # Create sidebar for mode selection
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=80)
        st.title("Navigation")
        app_mode = st.radio("Select Mode", ["Elle 💙 Mental Health", "My Knowledge Base"])
        
        # Status of prompts loading
        if not prompts_loaded:
            st.warning("⚠️ Using default prompts as prompts.yaml could not be loaded")
        else:
            st.success("✓ Using prompts from prompts.yaml")
            
        # User ID for multi-user support
        st.info(f"Session ID: {user_id[:8]}...")
        
        # Add clear conversation button
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("Save Session"):
                if user_data["conversation_summary"]:
                    if save_conversation_summary(user_id, user_data["conversation_summary"], user_data["therapy_types"]):
                        st.success("Conversation saved!")
                    else:
                        st.error("Failed to save conversation")
                else:
                    st.warning("No conversation to save yet")
                
        with clear_col:
            if st.button("Clear Chat"):
                # Save summary before clearing if it exists
                if user_data["conversation_summary"] and len(user_data["messages"]) > 3:
                    save_conversation_summary(user_id, user_data["conversation_summary"], user_data["therapy_types"])
                
                user_data["messages"] = []
                user_data["conversation_summary"] = ""
                user_data["therapy_types"] = []
                st.rerun()
            
        # Speech input
        speech_col1, speech_col2 = st.columns(2)
        with speech_col1:
            speech_input = st.button("🎤 Speech Input")
        with speech_col2:
            tts_enabled = st.checkbox("🔊 TTS Output")
            
        # Other settings
        st.subheader("Settings")
        image_gen = st.checkbox("Generate images", value=True)
        
        # About
        with st.expander("About Elle"):
            st.write("""
            **Elle 💙** is a mental health AI assistant that adapts its approach based on your emotional needs.
            
            It provides different types of support:
            - General emotional support and empathy
            - Crisis intervention for suicidal thoughts
            - Anger management techniques
            - Motivation for depression
            - Dialectical Behavior Therapy (DBT)
            - Cognitive Behavioral Therapy (CBT)
            
            Your conversations are private and stored only in your browser session.
            
            Created by Joel Bansal - IIT Kanpur 
            """)
    
    if app_mode == "Elle 💙 Mental Health":
        # Use container for better styling and layout
        main_container = st.container()
        with main_container:
            # Custom styled header
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.title("Elle 💙 - Your Mental Health Companion")
            st.markdown('<p style="color: #616161; margin-bottom: 2rem;">A caring space for emotional support whenever you need it.</p>', unsafe_allow_html=True)
            
            # Chat history container
            chat_container = st.container()
            
            # Check for speech input first
            text_from_speech = None
            if speech_input:
                try:
                    with st.spinner("Listening..."):
                        text_from_speech = recognize_speech()
                    if text_from_speech:
                        st.success(f"Heard: {text_from_speech}")
                except Exception as e:
                    st.error(f"Speech recognition failed: {str(e)}")
            
            # Chat input
            prompt = st.chat_input("Type your message here...") or text_from_speech
            
            # Display chat history with custom styling
            with chat_container:
                if not user_data["messages"]:
                    st.markdown('<div style="text-align: center; padding: 3rem; color: #9e9e9e;">👋 Hello! How are you feeling today? I\'m Elle, your mental health companion.</div>', unsafe_allow_html=True)
                
                for message in user_data["messages"]:
                    # Check if message contains therapy type
                    therapy_type = message.get("therapy_type", None)
                    display_message(message["role"], message["content"], therapy_type)
            
            if prompt:
                # Add user message to chat history
                user_data["messages"].append({
                    "role": "user", 
                    "content": prompt,
                    "timestamp": time.time()
                })
                
                # Display user message with custom styling
                display_message("user", prompt)
                
                # Generate response
                with st.spinner("Elle is typing..."):
                    # Determine which therapy approach to use based on content
                    text_lower = prompt.lower()
                    
                    # Select appropriate prompt based on content
                    if any(word in text_lower for word in ["suicide", "kill myself", "end my life", "don't want to live"]):
                        system_prompt = prompts["suicide_prevention_agent_prompt"]
                        therapy_type = "Suicide Prevention"
                    elif any(word in text_lower for word in ["angry", "furious", "rage", "hate", "mad"]):
                        system_prompt = prompts["anger_prevention_agent_prompt"]
                        therapy_type = "Anger Management"
                    elif any(word in text_lower for word in ["sad", "depressed", "hopeless", "unmotivated", "no point"]):
                        system_prompt = prompts["motivational_agent_prompt"]
                        therapy_type = "Motivation & Depression Support"
                    elif any(word in text_lower for word in ["overwhelmed", "emotions", "feeling too much"]):
                        system_prompt = prompts["dbt_agent_prompt"]
                        therapy_type = "Dialectical Behavior Therapy"
                    elif any(word in text_lower for word in ["negative thoughts", "always think", "can't stop thinking"]):
                        system_prompt = prompts["cbt_agent_prompt"]
                        therapy_type = "Cognitive Behavioral Therapy"
                    else:
                        system_prompt = prompts["empathetic_agent_prompt"]
                        therapy_type = "General Emotional Support"
                    
                    # Add therapy type to the tracking list
                    user_data["therapy_types"].append(therapy_type)
                    
                    # Get conversation summary for context if we have enough messages
                    if len(user_data["messages"]) >= 3:
                        # Generate or update summary every few messages
                        if len(user_data["messages"]) % 3 == 0:
                            user_data["conversation_summary"] = generate_conversation_summary(user_data["messages"])
                            
                            # Save summary to CSV file automatically
                            save_conversation_summary(user_id, user_data["conversation_summary"], user_data["therapy_types"])
                    
                    # Include the summary in the prompt if available
                    context = ""
                    if user_data["conversation_summary"]:
                        context = f"CONVERSATION SUMMARY SO FAR:\n{user_data['conversation_summary']}\n\n"
                    
                    # Generate response using Gemini
                    try:
                        model = genai.GenerativeModel("gemini-1.5-pro")
                        full_prompt = f"{system_prompt}\n\n{context}User: {prompt}\n\nAssistant:"
                        
                        response = model.generate_content(full_prompt)
                        response_text = response.text
                        
                        # Create multimodal response with text and possibly an image
                        multimodal_response = {"text": response_text}
                        
                        # Add image if enabled
                        if image_gen and any(trigger in text_lower for trigger in ["feel", "stress", "anxiety", "help", "relax", "image"]):
                            img_url = generate_mental_health_image(therapy_type)
                            multimodal_response["image_url"] = img_url
                        
                        # Use text-to-speech if enabled
                        if tts_enabled:
                            thread = threading.Thread(target=speak_text, args=(response_text,))
                            thread.start()
                        
                        # Display response with custom styling
                        display_message("assistant", multimodal_response, therapy_type)
                        
                        # Add response to chat history
                        user_data["messages"].append({
                            "role": "assistant", 
                            "content": multimodal_response,
                            "therapy_type": therapy_type,
                            "timestamp": time.time()
                        })
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    elif app_mode == "My Knowledge Base":
        st.title("Personal Knowledge Base")
        st.write("Upload documents to ask questions about them.")

        # Upload PDF
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    document_text = extract_text_from_pdf(uploaded_file)
                    document_chunks = split_text_into_chunks(document_text)
                    chunk_embeddings = np.array([get_embedding(chunk) for chunk in document_chunks])
                    user_data["chunks"] = document_chunks
                    user_data["chunk_embeddings"] = chunk_embeddings
                    st.success("Document processed successfully!")
                    
                    # Show document stats
                    # st.info(f"Document processed: {len(document_chunks)} chunks, {len(document_text)} characters")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        if user_data["chunks"] is not None and user_data["chunk_embeddings"] is not None:
            st.subheader("Ask a question")
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_area("🔍 Ask a question about this document:", "")
            with col2:
                search_btn = st.button("Search")
                
            if question and search_btn:
                with st.spinner("Searching for answer..."):
                    try:
                        relevant_chunks = retrieve_relevant_chunks(
                            question, 
                            user_data["chunks"], 
                            user_data["chunk_embeddings"]
                        )
                        combined_chunks = " ".join(relevant_chunks)
                        
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        response = model.generate_content(
                            f"Based on the following text, answer the question thoroughly and accurately: {combined_chunks}\n\nQuestion: {question}"
                        )
                        
                        st.markdown("### Answer:")
                        st.markdown(f"""
                        <div style="background-color: #1D1F27; color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #6c5ce7;">
                        {response.text}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Optional view sources button - hidden by default
                        with st.expander("Show source information", expanded=False):
                            st.info(f"Answer generated from {len(relevant_chunks)} relevant sections of your document")
                            # Show sources toggle
                            if st.checkbox("View detailed source chunks"):
                                for i, chunk in enumerate(relevant_chunks):
                                    st.markdown(f"**Chunk {i+1}**")
                                    st.markdown(f"<div style='max-height: 200px; overflow-y: auto; padding: 10px;background-color:#1D1F27; color: white; border-radius: 5px; margin-bottom: 10px; border: 1px solid #e0e0e0;'>{chunk[:500]}...</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error searching document: {str(e)}")

if __name__ == "__main__":
    main()