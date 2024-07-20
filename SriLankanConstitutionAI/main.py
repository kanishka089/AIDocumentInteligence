import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import os

#instructions to start
#https://www.linkedin.com/pulse/enhance-document-management-ai-extract-insights-from-pdfs-le-sueur-kfd5f/
#https://github.com/RexiaAI/codeExamples/blob/main/localRAG/RAG.py
#ollama pull nomic-embed-text

# Load, split, and retrieve documents from a local PDF file
def loadAndRetrieveDocuments() -> Chroma:
    loader = pdf.PyPDFLoader("constitution.pdf")
    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentSplits = textSplitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorStore = Chroma.from_documents(documents=documentSplits, embedding=embeddings)
    return vectorStore.as_retriever()

# Format a list of documents into a string
def formatDocuments(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)

# Define the RAG chain function
def ragChain(question: str) -> str:
    retriever = loadAndRetrieveDocuments()
    retrievedDocuments = retriever.invoke(question)
    formattedContext = formatDocuments(retrievedDocuments)
    formattedPrompt = f"Question: {question}\n\nContext: {formattedContext}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formattedPrompt}])
    return response['message']['content']

# Gradio interface
interface = gr.Interface(
    fn=ragChain,
    inputs="text",
    outputs="text",
    title="Q & A on Sri Lankan Constitution"
)

# Launch the app
interface.launch()