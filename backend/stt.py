import os
import requests
from dotenv import load_dotenv

# ✅ Load API key
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

class ElevenLabsTranscriber:
    def __init__(self):
        self.api_key = ELEVENLABS_API_KEY
        self.base_url = "https://api.elevenlabs.io/v1/speech-to-text"
        self.model_id = "scribe_v1"

    def transcribe(self, file_path: str):
        """Send audio file to ElevenLabs and return transcription."""
        try:
            with open(file_path, 'rb') as audio_file:
                response = requests.post(
                    self.base_url,
                    headers={"xi-api-key": self.api_key},
                    data={
                        "model_id": self.model_id,
                        "file_format": "other",
                    },
                    files={"file": (os.path.basename(file_path), audio_file)},
                )

            result = response.json()
            if "text" in result:
                return result["text"]
            else:
                return {"error": result}

        except Exception as e:
            return {"error": str(e)}