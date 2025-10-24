from stt import ElevenLabsTranscriber
from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File, HTTPException
import os

api = APIRouter(prefix="/api")


@api.get("/hello")
def hello():
    return {"message": "Hello, Anon!"}


@api.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and get back the transcription text.
    """
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            buffer.write(await file.read())

        # Use transcriber from stt.py
        transcriber = ElevenLabsTranscriber()
        result = transcriber.transcribe(temp_file)

        # Cleanup
        os.remove(temp_file)

        if isinstance(result, str):
            return {"text": result}
        else:
            raise HTTPException(status_code=400, detail=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))