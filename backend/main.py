from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
import dotenv
import os
from pydantic import BaseModel

class Prompt(BaseModel):
    prompt: str

app = FastAPI()
dotenv.load_dotenv()

print(os.environ.get("BASE_URL"))
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate")
async def generate(prompt: Prompt):
    client = OpenAI(base_url="http://0.0.0.0:8080", api_key="sk-1234")
    message = [
        {
            "role": "system",
            "content": "Hello, I'm a chatbot that can help you with anything you need. How can I help you today?"
        },
        {
            "role": "user",
            "content": prompt.prompt
        }
    ]
    
    response = client.chat.completions.create(
        model="",
        messages=message)

    return response.choices[0].message.content
    
# 404 page if the user tries to access a page that doesn't exist
@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str):
    raise HTTPException(status_code=404, detail="Page not found")
