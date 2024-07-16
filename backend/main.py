from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from openai import OpenAI
import dotenv
import os
from pydantic import BaseModel
from langchain_community.document_loaders.pdf import PyPDFLoader
from contextlib import asynccontextmanager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import torch
from langchain import hub
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

chat_model = ChatOpenAI(base_url=os.getenv("BASE_URL"), api_key="sk-1234")

class Prompt(BaseModel):
    prompt: str

app = FastAPI()


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


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    split = RecursiveCharacterTextSplitter.from_huggingface_tokenizer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",chunk_size=200)
    with open("file.pdf", "wb") as f:
        f.write(file.file.read())
    doc = PyPDFLoader("file.pdf").load_and_split(text_splitter=split)
    db = Chroma.from_documents(doc, embeddings_model,persist_directory="embeddings")
    os.remove("file.pdf")
    return {"message": "PDF uploaded and processed successfully."}

@app.post("/similarity_search")
async def similarity_search(prompt: Prompt):
    query = prompt.prompt
    
    
    db = Chroma(persist_directory="embeddings", embedding_function=embeddings_model)
    #results = db.similarity_search(query, k=5)
    #results = [item.page_content for item in results]
    retriever = db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
    )

    result = rag_chain.invoke(query)

    return result


# 404 page if the user tries to access a page that doesn't exist
@app.get("/{path:path}", response_class=HTMLResponse)
async def catch_all(path: str):
    raise HTTPException(status_code=404, detail="Page not found")

