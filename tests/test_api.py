import os
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_transcribe_valid_file():
    file_path = "tests/audio.wav"
    
    with open(file_path, "rb") as file:
        response = requests.post(f"{BASE_URL}/transcribe", files={"file": file})
        
    assert response.status_code == 200
    assert "text" in response.json()

def test_transcribe_invalid_file():
    file_path = "tests/invalid_file.txt"
    
    with open(file_path, "rb") as file:
        response = requests.post(f"{BASE_URL}/transcribe", files={"file": file})
        
    assert response.status_code == 400

def test_transcribe_no_file():
    response = requests.post(f"{BASE_URL}/transcribe")
    assert response.status_code == 422