from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

# Инициализация FastAPI и модели
app = FastAPI()
model_name = "gpt2"  # Используем GPT-2, или можете заменить на вашу модель
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Проверка, если у токенизатора нет токена паддинга, то добавляем его
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Используем токен окончания как токен паддинга

model = AutoModelForCausalLM.from_pretrained(model_name)

@app.post("/generate")
async def generate_text(prompt: str):
    # Токенизация с паддингом
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Генерация текста
    outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"generated_text": generated_text}