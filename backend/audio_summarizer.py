import os
import io
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter

# Load environment variables
load_dotenv()

class AudioSummarizer:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")  # Default voice if not set

    @staticmethod
    def high_pass_filter(audio_data: np.ndarray, sample_rate: int, cutoff=80):
        """Apply a high-pass filter to remove low-frequency noise."""
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(1, normal_cutoff, btype='high')
        return lfilter(b, a, audio_data)

    @staticmethod
    def apply_filter_and_save_audio(mp3_bytes: bytes, output_file: str):
        """Apply noise filter and save the MP3 file."""
        audio_segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        filtered_samples = AudioSummarizer.high_pass_filter(samples, audio_segment.frame_rate)
        filtered_audio = AudioSegment(
            filtered_samples.astype(np.int16).tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=2,
            channels=1
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        filtered_audio.export(output_file, format="mp3")
        print(f"✅ Filtered audio saved as {output_file}")

    def summarize_text(self, text: str) -> str:
        """Summarize input text using GPT."""
        prompt = f"Summarize the following text in concise and clear sentences:\n\n{text}"
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    def generate_audio_from_text(self, text: str, output_file: str):
        """Convert summarized text to speech using ElevenLabs."""
        audio_data = self.elevenlabs_client.text_to_speech.convert(
            voice_id=self.voice_id,
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

    def summarize_and_generate_audio(self, input_text: str, output_audio: str):
        """Full workflow: summarize text -> TTS -> filter."""
        print("📝 Summarizing input text...")
        summary = self.summarize_text(input_text)
        print(f"\n📄 Summary:\n{summary}\n")
        
        print("🔊 Generating audio summary...")
        self.generate_audio_from_text(summary, output_audio)
        print("✅ Audio summary saved successfully!")

# Example usage
if __name__ == "__main__":
    summarizer = AudioSummarizer()
    input_text = """
    In today’s meeting, the team discussed project progress and upcoming deadlines. 
    Key action items include finalizing the UI design, testing backend APIs, 
    and preparing the product demo for next week’s client review.
    """
    output_file = "output/meeting_summary.mp3"
    summarizer.summarize_and_generate_audio(input_text, output_file)
