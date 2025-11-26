import streamlit as st
import os
import time
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from openai import OpenAI
from utils.audio_utils import AudioProcessor
from utils.llm_utils import LLMProcessor
import sounddevice as sd
import soundfile as sf
from utils.audio_utils import play_audio_local

# Load environment variables
load_dotenv()

# Configure OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set OPENAI_API_KEY in .env file")
else:
            # Check API quota
    try:
        client = OpenAI()
        # Try a small API call to check quota
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
    except Exception as e:
        if "insufficient_quota" in str(e):
            st.error("""
            ⚠️ OpenAI API quota exceeded!

            Please check your API usage and billing details at:
            https://platform.openai.com/account/billing/overview
            
            You can:
            1. Add payment method to get more quota
            2. Wait for next billing cycle
            3. Create new account for free credits
            
            See README.md troubleshooting section for details.
            """)
        else:
            st.error(f"API verification error: {str(e)}")

# Configuration parameters
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# Initialize processors
audio_processor = AudioProcessor(sample_rate=SAMPLE_RATE)
llm_processor = LLMProcessor(model_name=LLM_MODEL)

# Initialize Streamlit state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recording_duration" not in st.session_state:
    st.session_state.recording_duration = 5
if "mode" not in st.session_state:
    st.session_state.mode = "customer_service"  # Default to customer service mode

def visualize_audio(audio_path):
    """Visualize audio waveform"""
    try:
        import soundfile as sf
        data, _ = sf.read(audio_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(np.arange(len(data)) / SAMPLE_RATE, data)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display image
        st.image(buf)
    except Exception as e:
        st.warning(f"Unable to visualize audio: {str(e)}")

def play_audio_local(filepath: str):
    data, sr = sf.read(filepath, dtype="float32")
    sd.play(data, sr)
    sd.wait()

def handle_audio_input():
    """Handle audio input"""
    with st.spinner(f"Recording {st.session_state.recording_duration} seconds..."):
        audio_file = audio_processor.record_audio(duration=st.session_state.recording_duration)
        if audio_file:
            st.audio(audio_file)
            visualize_audio(audio_file)
            
            # Transcribe audio
            with st.spinner("Transcribing..."):
                transcript = audio_processor.transcribe_audio(audio_file)
                if transcript:
                    st.info(f"Transcription: {transcript}")
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = llm_processor.generate_response(
                            transcript, 
                            conversation_history=st.session_state.messages
                        )
                        
                        # Update conversation history
                        st.session_state.messages.append({"role": "user", "content": transcript})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Generate voice response
                        audio_response = audio_processor.text_to_speech(response)
                        if audio_response:
                            st.success("AI response:")
                            st.write(response)
                            st.audio(audio_response)
                            play_audio_local(audio_response)
                else:
                    st.error("Audio transcription failed")
        else:
            st.error("Recording failed")

def handle_text_input():
    """Handle text input"""
    user_input = st.text_input("Enter your question:")
    if user_input:
        with st.spinner("Generating response..."):
            response = llm_processor.generate_response(
                user_input, 
                conversation_history=st.session_state.messages
            )
            
            # Update conversation history
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Generate voice response
            audio_response = audio_processor.text_to_speech(response)
            if audio_response:
                st.success("AI response:")
                st.write(response)
                st.audio(audio_response)
                
                # Reset input box
                st.rerun()

def display_conversation_history():
    """Display conversation history"""
    st.subheader("Conversation History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def settings_section():
    """Settings section"""
    st.sidebar.header("Settings")
    
    # Recording duration setting
    st.session_state.recording_duration = st.sidebar.slider(
        "Recording Duration (seconds)", 
        min_value=3, 
        max_value=15, 
        value=st.session_state.recording_duration
    )
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select mode",
        ["Customer Service", "Lead Generation"]
    )
    
    if mode == "Customer Service" and st.session_state.mode != "customer_service":
        st.session_state.mode = "customer_service"
        llm_processor.customize_for_call_center()
        st.sidebar.success("Switched to Customer Service mode")
    elif mode == "Lead Generation" and st.session_state.mode != "lead_generation":
        st.session_state.mode = "lead_generation"
        llm_processor.customize_for_lead_generation()
        st.sidebar.success("Switched to Lead Generation mode")
    
    # Clear conversation history
    if st.sidebar.button("Clear Conversation History"):
        st.session_state.messages = []
        st.sidebar.success("Conversation History Cleared")
        st.rerun()

def main():
    st.title("AI Voice Customer Service Assistant")
    st.write("This is an AI assistant that can engage in voice conversations and help answer your questions.")
    
    # Setup sidebar
    settings_section()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Voice Interaction", "Text Interaction", "Conversation Analysis"])
    
    with tab1:
        st.subheader("Voice Input")
        if st.button("Start Recording"):
            handle_audio_input()
        
        # Upload audio file
        uploaded_file = st.file_uploader("Or Upload Audio File", type=["wav", "mp3"])
        if uploaded_file:
            with st.spinner("Processing uploaded audio..."):
                # Save uploaded file
                with open("temp_upload.wav", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Transcribe audio
                transcript = audio_processor.transcribe_audio("temp_upload.wav")
                if transcript:
                    st.info(f"Transcription: {transcript}")
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = llm_processor.generate_response(
                            transcript, 
                            conversation_history=st.session_state.messages
                        )
                        
                        # Update conversation history
                        st.session_state.messages.append({"role": "user", "content": transcript})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Generate voice response
                        audio_response = audio_processor.text_to_speech(response)
                        if audio_response:
                            st.success("AI response:")
                            st.write(response)
                            st.audio(audio_response)
    
    with tab2:
        st.subheader("Text Interaction")
        handle_text_input()
    
    with tab3:
        st.subheader("Conversation Analysis")
        if st.session_state.messages:
            if st.button("Analyze Conversation"):
                with st.spinner("Analyzing conversation..."):
                    analysis = llm_processor.analyze_conversation(st.session_state.messages)
                    st.write(analysis["analysis"])
        else:
            st.info("Conversation history is empty, cannot perform analysis")
    
    # display history
    display_conversation_history()

if __name__ == "__main__":
    main()