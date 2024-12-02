import os
import getpass
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage , SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Use Gemini Embeddings
from langchain_google_vertexai import ChatVertexAI  # Chatbot model
from langchain import hub
from pathlib import Path

from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, GibberishText, DetectPII

guard = Guard().use_many(
    ToxicLanguage(on_fail=OnFailAction.EXCEPTION),
    GibberishText(on_fail=OnFailAction.EXCEPTION),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"],on_fail=OnFailAction.EXCEPTION),

)

# Set up the Google API key for authentication with Vertex AI (Gemini)
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
else:
    print("GEMINI_API_KEY is successfully loaded from the environment.")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini model (ChatVertexAI)
chat_model = ChatVertexAI(model="gemini-1.5-flash")  # Gemini model

# Paths and configuration
PDF_FOLDER = "./vendor_pdfs"
VECTOR_STORE_PATH = "./vendor_faiss_index"

# Load and preprocess PDFs
def load_pdfs(folder: str):
    pdf_files = Path(folder).glob("*.pdf")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        raw_documents = loader.load()
        documents.extend(text_splitter.split_documents(raw_documents))
    return documents

# Gemini Embeddings (GoogleGenerativeAIEmbeddings)
def initialize_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

# Initialize FAISS vector store using Gemini embeddings
def initialize_faiss_store(documents):
    embeddings = initialize_gemini_embeddings()  # Use Gemini embeddings for document processing
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

# Load or create FAISS vector store
if Path(VECTOR_STORE_PATH).exists():
    faiss_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        initialize_gemini_embeddings(),  # Use Gemini embeddings to load vector store
        allow_dangerous_deserialization=True,  # Security enhancement
    )
else:
    documents = load_pdfs(PDF_FOLDER)  # Load and preprocess documents
    faiss_store = initialize_faiss_store(documents)

# Create retrieval chain using FAISS and the QA prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    chat_model, retrieval_qa_chat_prompt  # Use ChatVertexAI for combining document-based Q&A
)

retrieval_chain = create_retrieval_chain(
    faiss_store.as_retriever(), combine_docs_chain
)

# Pydantic model for request
class ChatRequest(BaseModel):
    history: list[dict]  # Chat history as {"role": "human"/"ai", "content": "text"}
    message: str         # Current user message

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        
  

        
        
            # Initialize the chat history
        history = []

        # # Add a system message if the provided history is empty
        # if not request.history:
        system_message = SystemMessage(
    content="""
    You are a support chatbot for the Vendor Management System, which is part of the Onyx ERP System developed by Ultimate Solutions. Your main responsibility is to assist users with inquiries related to the ERP system, focusing on vendor management processes and tasks. Ensure that your responses are clear, concise, and accurate, maintaining a professional tone at all times.

    - For ERP-related queries, provide detailed instructions, troubleshooting steps, and guidance on managing vendor-related tasks.
    - For inquiries outside the ERP system or unrelated to vendor management, politely inform the user with: "Sorry, I cannot assist with that."

    Your goal is to help users with actionable information while keeping the conversation focused, courteous, and professional. If you are unsure of an answer or if the issue requires human assistance, kindly suggest reaching out to our support team at support@ultimatesolutions.com for further help.

    Thank you for helping to ensure a smooth and efficient user experience!
    """
)

        history.append(system_message)

        # Convert the request history to LangChain message format
        history.extend(
            HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
            for msg in request.history
        )

        try:
           
           
            print(request.message)
            guard.validate(request.message)
            # Get the AI response using the retrieval chain
            response = retrieval_chain.invoke({"input": request.message, "chat_history": history})
            print(response)
            response = response["answer"]
            guard.validate(response)

        except Exception as e:
            print(e)
            response = "Sorry I can't help with that"    


        # Append the AI's response to the chat history
        ai_response = {"role": "ai", "content": response}
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run the Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
