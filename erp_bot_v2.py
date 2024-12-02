import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Use Gemini Embeddings
from langchain_google_vertexai import ChatVertexAI  # Chatbot model
from langchain.tools import Tool, tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain import hub

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

# Tools for fetching vendor details by name
@tool("get_vendor_by_name")
def get_vendor_by_name(name: str) -> str:
    """
    Fetches vendor details for a given vendor name.
    :param name: The Name of the vendor to fetch details for.
    :return: Vendor details as a string.
    """
    # Mocked response (replace with actual API/database call)
    mocked_vendors = {
        "Ahmed Khaled": "ID 12345: Status - Active, Products - 3, Dept - $500.00",
        "Ali Ahmed": "ID 67890: Status - Active, Products - 1, Dept - $00.00",
        "Saeed Omar": "ID 54321: Status - Not Active, Products - 5, Dept - $1000.00",
    }
    return mocked_vendors.get(name, f"Vendor Name {name} not found. Please check and try again.")

# Define the tools for the agent
tools = [get_vendor_by_name]

# Initialize LangChain agent with the Gemini model
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

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

# Define tool for retrieving documents
@tool("get_document_retriever")
def get_document_retriever(query: str, history: list) -> str:
    """
    Fetches relevant documents based on the user's query using the FAISS retriever.
    :param query: The query from the user.
    :param history: The chat history.
    :return: The response from the document retriever chain.
    """
    result = retrieval_chain.invoke({"input": query, "chat_history": history})
    return result['answer']

# Update tools with document retriever tool
tools.append(get_document_retriever)

# Combine tools and retriever into an AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Pydantic model for request
class ChatRequest(BaseModel):
    history: list[dict]  # Chat history as {"role": "human"/"ai", "content": "text"}
    message: str         # Current user message

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the chat history
        history = []

        # Add a system message if the provided history is empty
        system_message = SystemMessage(
            content="""
            You are an ERP Helpdesk Bot designed to assist with the Vendor Management System created by Ultimate Solutions. Your primary role is to provide clear, concise, and accurate responses related to the ERP system, vendor management processes, and any relevant support queries. If the user asks anything outside of these topics, politely inform them that you cannot assist with unrelated inquiries.

            - For ERP-related questions, provide detailed instructions, troubleshooting steps, or guidance on managing vendor-related tasks.
            - For non-ERP queries, respond with: "Sorry, I can't help with that."

            For any support-related inquiries, refer users to the customer support email: support@ultimatesolutions.com.

            Your goal is to ensure users receive helpful, actionable information regarding the Vendor Management System while maintaining a professional and focused tone.
            """
        )
        history.append(system_message)

        # Convert the request history to LangChain message format
        history.extend(
            HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
            for msg in request.history
        )

        # Append the new user message
        history.append(HumanMessage(content=request.message))

        # Get the AI response using the agent_executor
        response = agent_executor.invoke({"input": request.message, "chat_history": history})

        # Append the AI's response to the chat history
        ai_response = {"role": "ai", "content": response['answer']}
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Run the Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
