import os
import io
import json
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from openai import OpenAI
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

class AudioAssistant:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")  # Default voice ID

    @staticmethod
    def high_pass_filter(audio_data: np.ndarray, sample_rate: int, cutoff=80):
        """Apply a high-pass filter to remove low-frequency noise."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='high')
        return lfilter(b, a, audio_data)

    @staticmethod
    def apply_filter_and_save_audio(mp3_bytes: bytes, output_file: str):
        """Convert MP3 bytes to waveform, apply filter, and save back as MP3."""
        audio_segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        filtered_samples = AudioAssistant.high_pass_filter(samples, audio_segment.frame_rate)
        filtered_audio = AudioSegment(
            filtered_samples.astype(np.int16).tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=2,
            channels=1
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        filtered_audio.export(output_file, format="mp3")
        print(f"✅ Filtered audio saved as {output_file}")

    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using ElevenLabs."""
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        transcription = self.elevenlabs_client.audio.transcribe(audio_bytes)
        return transcription["text"]

    def summarize_text(self, text: str) -> str:
        """Summarize text using OpenAI GPT."""
        prompt = f"Please summarize the following text in concise, clear sentences:\n\n{text}"
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary

    def generate_audio_from_text(self, text: str, output_file: str):
        """Convert text to speech using ElevenLabs and save filtered audio."""
        audio_data = self.elevenlabs_client.text_to_speech.convert(
            #voice_id=self.voice_id,
            voice_id="Rachel",
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "use_speaker_boost": True,
                "similarity_boost": 1.0,
                "style": 1.0,
                "speed": 0.9
            }
        )
        audio_bytes = b''.join(chunk for chunk in audio_data if chunk)
        self.apply_filter_and_save_audio(audio_bytes, output_file)

    def process_audio_file(self, input_audio: str, output_audio: str):
        """Full workflow: transcribe -> summarize -> TTS -> filter."""
        print(f"📄 Transcribing {input_audio}...")
        transcription = self.transcribe_audio(input_audio)
        print(f"📝 Transcription:\n{transcription}\n")
        
        print("🔹 Summarizing text...")
        summary = self.summarize_text(transcription)
        print(f"📝 Summary:\n{summary}\n")
        
        print("🔊 Generating audio from summary...")
        self.generate_audio_from_text(summary, output_audio)
        print("✅ Done!")

# Example usage
if __name__ == "__main__":
    assistant = AudioAssistant()
    input_file = "input/audio_input.mp3"
    output_file = "output/audio_summary.mp3"
    assistant.process_audio_file(input_file, output_file)
