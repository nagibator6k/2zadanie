from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Загрузка моделей
model_1 = AutoModelForCausalLM.from_pretrained("distilgpt2", device_map="cpu")
tokenizer_1 = AutoTokenizer.from_pretrained("distilgpt2")

model_2 = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", device_map="cpu")
tokenizer_2 = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

generator_1 = pipeline('text-generation', model=model_1, tokenizer=tokenizer_1)
generator_2 = pipeline('text-generation', model=model_2, tokenizer=tokenizer_2)

# Тестирование модели на генерацию текста
prompt = "Что такое искусственный интеллект?"
response_1 = generator_1(prompt, max_length=100)
response_2 = generator_2(prompt, max_length=100)

print(f"Model 1 response: {response_1[0]['generated_text']}")
print(f"Model 2 response: {response_2[0]['generated_text']}")