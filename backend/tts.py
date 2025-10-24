import os
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load environment variables from .env file
load_dotenv()

class TextToSpeechGenerator:
    def __init__(self, api_key: str = None, voice_id: str = None):
        """
        Initialize ElevenLabs client and voice configuration.
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

        if not self.api_key:
            raise ValueError("❌ ELEVENLABS_API_KEY not found in environment or arguments.")
        
        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=self.api_key, base_url="https://api.elevenlabs.io/")

    def text_to_speech(self, text: str, output_path: str):
        """
        Convert text to speech using ElevenLabs API and save as MP3 file.
        """
        print("🔊 Generating speech...")

        response_stream = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.9,
                "style": 1.0,
                "use_speaker_boost": True
            }
        )

        # Join audio chunks
        audio_bytes = b"".join(chunk for chunk in response_stream if chunk)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save MP3
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        print(f"✅ Audio saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    tts = TextToSpeechGenerator()
    sample_text = (
        "Christopher and Shivan discuss a project aimed at creating concise summaries of lengthy audio content "
        "like podcasts using AI technology. They plan to test the AI's summarization capabilities in a demo phase."
    )
    tts.text_to_speech(sample_text, "output/summary_audio.mp3")
