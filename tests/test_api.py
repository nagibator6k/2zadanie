import pytest
from fastapi import FastAPI
import httpx
from threading import Thread
from fastapi.testclient import TestClient
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '1')))

from main import app

def run_server():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

@pytest.fixture(scope="module", autouse=True)
def start_server():
    server_thread = Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    time.sleep(2)

    yield

def test_transcribe_valid_file():

    client = httpx.Client()

    file_path = "tests/audio.wav"
    
    with open(file_path, "rb") as file:
        response = client.post("http://127.0.0.1:8000/transcribe", files={"file": file})
        
    assert response.status_code == 200
    assert "text" in response.json()