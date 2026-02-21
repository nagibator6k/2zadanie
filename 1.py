from fastapi import FastAPI, UploadFile
import whisper

app = FastAPI()

model = whisper.load_model("small")

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with open("audio.wav", "wb") as f:
        f.write(await file.read())
    
    result = model.transcribe("audio.wav")
    return {"text": result["text"]}