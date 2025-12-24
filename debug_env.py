from dotenv import load_dotenv
import os

print("--- DEBUG START ---")
# Пытаемся загрузить .env
loaded = load_dotenv(verbose=True)
print(f".env loaded status: {loaded}")

# Проверяем наличие файла
if os.path.exists(".env"):
    print("File .env exists.")
else:
    print("File .env NOT found in current directory.")

# Проверяем ключи (безопасно)
key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"OPENAI_API_KEY found: {key[:5]}...{key[-3:] if len(key) > 5 else ''}")
else:
    print("OPENAI_API_KEY NOT found in env variables.")

# Выводим все ключи которые есть (только имена)
print("Available keys in env:", [k for k in os.environ.keys() if "KEY" in k or "TOKEN" in k])
print("--- DEBUG END ---")
