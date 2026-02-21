from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Инициализация FastAPI и загрузка модели
app = FastAPI()

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Добавляем токен паддинга (используем EOS токен как паддинг)
tokenizer.pad_token = tokenizer.eos_token

class RequestModel(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    no_repeat_ngram_size: int = 2

@app.post("/generate")
async def generate_text(request: RequestModel):
    # Токенизация входного текста
    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Генерация текста
    outputs = model.generate(
        inputs["input_ids"],
        max_length=request.max_tokens + len(inputs["input_ids"][0]),
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        no_repeat_ngram_size=request.no_repeat_ngram_size,
        pad_token_id=model.config.pad_token_id,  # Указываем pad_token_id
        eos_token_id=model.config.eos_token_id,
        early_stopping=True
    )
    
    # Декодируем и возвращаем сгенерированный текст
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}