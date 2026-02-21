from fastapi import FastAPI, UploadFile
import whisper

app = FastAPI()

model = whisper.load_model("small")

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with open("audio.wav", "wb") as f:
        f.write(await file.read())
    
    result = model.transcribe("audio.wav")
    return {"text": result["text"]}