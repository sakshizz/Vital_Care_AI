# test_api.py
import google.generativeai as genai
from api_key import api_key

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')
response = model.generate_content("Say hello")
print(response.text)