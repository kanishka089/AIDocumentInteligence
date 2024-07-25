import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from huggingface_hub import InferenceClient
import requests
import urllib3
from dotenv import load_dotenv
import os
from nomic import embed

# instructions to start
# https://www.linkedin.com/pulse/enhance-document-management-ai-extract-insights-from-pdfs-le-sueur-kfd5f/
# https://github.com/RexiaAI/codeExamples/blob/main/localRAG/RAG.py
# ollama pull nomic-embed-text
load_dotenv('secret.env')#remove string if hosting in huggingface
token = os.getenv('HUGGINGFACE_TOKEN')
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=token
)

# Load, split, and retrieve documents from a local PDF file
def loadAndRetrieveDocuments() -> Chroma:
    loader = pdf.PyPDFLoader("k.pdf")#constitution
    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentSplits = textSplitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorStore = Chroma.from_documents(documents=documentSplits, embedding=embeddings)
    return vectorStore.as_retriever()


# Format a list of documents into a string
def formatDocuments(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)


retriever = loadAndRetrieveDocuments()

# Chat history
chat_history = []


# Define the RAG chain function
def ragChain(question: str, include_history: bool) -> str:
    global chat_history
    retrievedDocuments = retriever.invoke(question)
    formattedContext = formatDocuments(retrievedDocuments)
    formattedPrompt = f"Question: {question}\n\nContext: {formattedContext}"

    # Prepare messages with or without history based on checkbox state
    if include_history:
        messages = chat_history + [{"role": "user", "content": formattedPrompt}]
    else:
        messages = [{"role": "user", "content": formattedPrompt}]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=False
        )
        # Extract the generated response text using dataclass attributes
        generated_text = ""
        if response and response.choices:
            generated_text = response.choices[0].message.content

        # Update chat history if including history
        if include_history:
            chat_history.append({"role": "user", "content": formattedPrompt})
            chat_history.append({"role": "assistant", "content": generated_text})

        return generated_text or "No response generated"

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


# Gradio interface
interface = gr.Interface(
    fn=ragChain,
    inputs=[
        gr.Textbox(label="Question"),
        gr.Checkbox(label="Include Chat History", value=True)
    ],
    outputs="text",
    title="Q & A on Sri Lankan Constitution"
)

# Launch the app
interface.launch()
