from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from openai import OpenAI
import dotenv
import os
from pydantic import BaseModel
from langchain_community.document_loaders.pdf import PyPDFLoader
from contextlib import asynccontextmanager
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
    client = OpenAI(base_url=os.getenv("BASE_URL"), api_key="sk-1234")
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

@asynccontextmanager
async def lifespan(app:FastAPI):
    yield
    os.remove("file.pdf")


@app.post("/upload")
#must be pdf
async def upload(file: UploadFile = File(...)):
    with open("file.pdf", "wb") as f:
        f.write(file.file.read())
    #pdf = PdfReader("file.pdf")
    #text = str(len(pdf.pages)) + " pages\n"
    #for page in pdf.pages:
    #    text += page.extract_text()
    doc = PyPDFLoader("file.pdf").load_and_split()
    text = ""
    for page in doc:
        text += page.page_content
    return text


# 404 page if the user tries to access a page that doesn't exist
@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str):
    raise HTTPException(status_code=404, detail="Page not found")

