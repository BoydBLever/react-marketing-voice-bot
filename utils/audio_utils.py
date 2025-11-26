import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from gtts import gTTS
import os
from typing import Optional
from utils.omni_asr import transcribe_with_omni_asr

def play_audio_local(path: str):
    """Play an audio file locally through the system speakers."""
    try:
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"[Audio Playback Error] {e}")

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Sampling rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self.temp_dir = tempfile.mkdtemp()

    def record_audio(self, duration: int = 5) -> Optional[str]:
        """
        Record audio
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Path to the recorded file
        """
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1
            )
            sd.wait()  # Wait for recording to complete
            
            # Save the recorded audio
            temp_path = os.path.join(self.temp_dir, 'recording.wav')
            sf.write(temp_path, recording, self.sample_rate)
            
            return temp_path
        except Exception as e:
            print(f"Recording error: {str(e)}")
            return None

    def transcribe_audio(self, audio_file):
        """
        Transcription function using OmniASR.
        """
        if isinstance(audio_file, str):
            return transcribe_with_omni_asr(audio_file)
        
        temp_path = "temp_input.wav"

        # save uploaded file to disk
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        
        return transcribe_with_omni_asr(temp_path)


    def text_to_speech(self, text: str, lang: str = 'zh') -> Optional[str]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            lang: Language code
            
        Returns:
            Path to generated audio file
        """
        try:
            tts = gTTS(text=text, lang=lang)
            temp_path = os.path.join(self.temp_dir, 'response.mp3')
            tts.save(temp_path)
            return temp_path
        except Exception as e:
            print(f"Text-to-speech error: {str(e)}")
            return None

    def preprocess_audio(self, audio_path: str) -> Optional[str]:
        """
        Preprocess audio file (resampling, noise reduction, etc.)
        
        Args:
            audio_path: Input audio file path
            
        Returns:
            Path to processed audio file
        """
        try:
            # Read audio file
            data, sample_rate = sf.read(audio_path)
            
            # Convert stereo to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample to target sample rate
            if sample_rate != self.sample_rate:
                # Resampling logic can be added here
                pass
            
            # Save processed audio
            processed_path = os.path.join(self.temp_dir, 'processed_audio.wav')
            sf.write(processed_path, data, self.sample_rate)
            
            return processed_path
        except Exception as e:
            print(f"Audio preprocessing error: {str(e)}")
            return None

    def cleanup(self):
        """
        Clean up temporary files
        """
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning temporary files: {str(e)}")
            