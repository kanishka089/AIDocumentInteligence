import os
import requests
import tempfile
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import pdf
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.embeddings import GPT4AllEmbeddings

load_dotenv('secret.env')  # remove string if hosting in huggingface
token = os.getenv('HUGGINGFACE_TOKEN')
client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=token,
)

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'false'}


# Function to download the PDF from a URL and load documents
def loadAndRetrieveDocuments(url: str, local_file_path: str) -> VectorStoreRetriever:
    try:
        # Attempt to download PDF
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses

        # Save PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_pdf_path = temp_file.name

    except Exception as e:
        print(f"Failed to download PDF from URL: {e}")
        # Use local file if URL download fails
        temp_pdf_path = local_file_path

    # Load the PDF from the temporary file
    loader = pdf.PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Clean up temporary file if created
    if temp_pdf_path != local_file_path:
        os.remove(temp_pdf_path)

    # Process documents
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentSplits = textSplitter.split_documents(documents)
    vectorStore = Chroma.from_documents(documents=documentSplits, embedding=GPT4AllEmbeddings(model_name=model_name,
                                                                                              gpt4all_kwargs=gpt4all_kwargs))
    return vectorStore.as_retriever()


def formatDocuments(documents: list) -> str:
    return "\n\n".join(document.page_content for document in documents)


# Define URL and local file path
url = "http://www.parliament.lk/files/pdf/constitution.pdf"
local_file_path = "constitution.pdf"  # Local file path

# Load documents from URL or local file
retriever = loadAndRetrieveDocuments(url, local_file_path)

# Chat history
chat_history = []


def ragChain(question: str) -> str:
    global chat_history
    retrievedDocuments = retriever.invoke(question)
    formattedContext = formatDocuments(retrievedDocuments)
    formattedPrompt = (f"Question: {question}\n\n"
                       f"Context: {formattedContext}\n\n"
                       f"Please provide a detailed answer based solely on the provided context.")

    messages = chat_history + [{"role": "user", "content": formattedPrompt}]

    response = client.chat_completion(
        messages=messages,
        max_tokens=700,
        stream=False
    )
    # Extract the generated response text using dataclass attributes
    generated_text = ""
    if response and response.choices:
        generated_text = response.choices[0].message.content

    # Update chat history
    chat_history.append({"role": "user", "content": formattedPrompt})
    chat_history.append({"role": "assistant", "content": generated_text})

    return generated_text or "No response generated"

# Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            textbox = gr.Textbox(label="Question")
            with gr.Row():
                buttonTerms = gr.Button("Terms of use")
                button = gr.Button("Submit")

        with gr.Column():
            output = gr.Textbox(label="Output", lines=25)


    def on_button_click(question):
        # Call the ragChain function with the question
        answer = ragChain(question)
        return answer

    def on_term_button_click():
        return ("The information provided by this application is generated using advanced technologies, including "
                "natural language processing models, document retrieval systems, and embeddings-based search "
                "algorithms. While these technologies are designed to offer accurate and relevant information, "
                "they may not always be up-to-date or fully accurate.The owner of this application does not accept "
                "any responsibility for potential inaccuracies, misleading information, or any consequences that may "
                "arise from the use of the application. Users are encouraged to verify the information independently "
                "and consult additional sources when making decisions based on the information provided by this app.")


    # Bind the button to the function
    button.click(on_button_click, inputs=textbox, outputs=output)
    buttonTerms.click(on_term_button_click, outputs=output)


demo.launch()
