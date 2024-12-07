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
import json

from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage, GibberishText, DetectPII


guard = Guard().use_many(
    ToxicLanguage(on_fail=OnFailAction.EXCEPTION),
   # GibberishText(validation_method="sentence" , threshold=1.0  , on_fail=OnFailAction.EXCEPTION),
   # DetectPII(on_fail=OnFailAction.EXCEPTION),

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
chat_model = ChatVertexAI(model="gemini-1.5-flash"  ,  temperature=0.2)  # Gemini model

# Tools for fetching vendor details by name
import json

@tool("get_vendor_by_name")
def get_vendor_by_name(name: str) -> str:
    """
    Fetches Onyx Vendor Management System vendor information for a given vendor name from JSON file.
    :param name: The name of the vendor to fetch details for.
    :return: Vendor details as a string.
    """
    # Load the JSON file containing vendor data
    try:
        with open('mock_vendors.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Retrieve vendor details from the mock data
        vendor_info = data["vendors"].get(name, None)
        
        if vendor_info:
            # Format the response with all relevant information
            payment_history = "\n".join([f"  {entry['date']} - ${entry['amount']:.2f} ({entry['status']})" for entry in vendor_info['payment_history']])
            return (
                f"ID {vendor_info['id']}: Status - {vendor_info['status']}, Products - {vendor_info['products']}, "
                f"Dept - ${vendor_info['dept']:.2f}, Business Type - {vendor_info['business_type']}, "
                f"Contact - Phone: {vendor_info['contact']['phone']}, Email: {vendor_info['contact']['email']}, "
                f"Contract Status - {vendor_info['contract_status']}, Rating - {vendor_info['rating']}\n"
                f"Payment History:\n{payment_history}"
            )
        else:
            return f"Vendor Name '{name}' not found. Please check and try again."
    
    except FileNotFoundError:
        return "Error: The vendor data file is missing."
    except json.JSONDecodeError:
        return "Error: Failed to parse vendor data from the file."
@tool("get_all_vendors_info")
def get_all_vendors_info(no_parmas: str) -> str:
    """
    Fetches all vendors' information from the Onyx Vendor Management System and returns
    the data as an HTML table to be rendered in a chat.
    
    :return: An HTML table string containing all vendor details.
    """
    try:
        # Load the JSON file containing vendor data
        with open('mock_vendors.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        vendors = data["vendors"]
        
        if not vendors:
            return "<p>No vendor data found.</p>"
        
        # Start building the HTML table
        table_html = """
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: left;">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Products</th>
                    <th>Debt</th>
                    <th>Business Type</th>
                    <th>Contract Status</th>
                    <th>Phone</th>
                    <th>Email</th>
                    <th>Rating</th>
                    <th>Payment History</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Iterate over the vendors and add rows to the table
        for name, vendor_info in vendors.items():
            payment_history = "\n".join([f"{entry['date']} - ${entry['amount']:.2f} ({entry['status']})"
                                        for entry in vendor_info['payment_history']])
            
            # Adding each vendor's details in the table row
            table_html += f"""
            <tr>
                <td>{name}</td>
                <td>{vendor_info['status']}</td>
                <td>{vendor_info['products']}</td>
                <td>${vendor_info['dept']:.2f}</td>
                <td>{vendor_info['business_type']}</td>
                <td>{vendor_info['contract_status']}</td>
                <td>{vendor_info['contact']['phone']}</td>
                <td>{vendor_info['contact']['email']}</td>
                <td>{vendor_info['rating']}</td>
                <td><pre>{payment_history}</pre></td>
            </tr>
            """
        
        # Close the table HTML
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    except FileNotFoundError:
        return "<p>Error: The vendor data file 'mock_vendors.json' is missing.</p>"
    except json.JSONDecodeError:
        return "<p>Error: Failed to parse the vendor data from the file.</p>"
    except Exception as e:
        return f"<p>An unexpected error occurred: {str(e)}</p>"

    

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

# Define tool for retrieving documents
@tool("search_onyx_vendor_system_docs")
def search_onyx_vendor_system_docs(query: str) -> str:
    """
    Fetches Onyx Vendor Management System relevant documents based on the user's query.
    :param query: A query String containing the query.
    :return: The response from Onyx Vendor Management System relevant documents.
    """

    result = retrieval_chain.invoke({"input": query})
    print("Document retrieval result:", result)
    return result['answer']

# Define the tools for the agent
tools = [search_onyx_vendor_system_docs, get_vendor_by_name , get_all_vendors_info]

# Initialize LangChain agent with the Gemini model
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
    
)



@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the chat history
        history = []

        # Add a system message if the provided history is empty
       # Add a system message if the provided history is empty
     # The system message defining the scope of your bot
        system_message = SystemMessage(
    content="""
        You are the **Onyx Vendor Management Assistant**, specifically designed to assist with the **Vendor Management System (VMS)** developed by **Ultimate Solutions**.

        Your role is to provide **clear**, **actionable**, and **focused** responses to queries related to vendor management within the VMS.

        **Introduction:**  
        Always introduce yourself as the **Vendor Management Assistant** when asked questions like "Can you introduce yourself?" or similar queries. For example:  
        **"Hello, I am the Vendor Management Assistant, here to help you with the Vendor Management System (VMS) developed by Ultimate Solutions."**

        **Scope of Responses:**  
        You should only provide answers related to the **Vendor Management System (VMS)**, including:
        - Managing vendor profiles (adding, editing, viewing details)
        - Answering queries about vendor status, product details, and related information
        - Providing instructions or troubleshooting for the VMS
        - Handling vendor inquiries by retrieving relevant data from the system

        **Out-of-Scope Queries:**  
        If a query is outside the scope of the VMS or the ERP system, respond with:  
        **"Sorry, I can't help with that. Please ask questions related to the Vendor Management System."**

        **Using Tools:**  
        1. Use **`get_vendor_by_name`** if the query is about a specific vendor. Example query: "What is the status of Ahmed Khaled?"
        2. Use **`get_all_vendors_info`** if the query is about retrieving information for all vendors. Example query: "Give me the vendor information."
        3 Use **`search_onyx_vendor_system_docs`** to retrieve specific vendor system-related data.
        
        If neither tool can provide an answer after **one attempt**, **stop retrying** and inform the user that the information is unavailable.

        If you are not using the tools, introduce yourself as the **Onyx Vendor Management Assistant** and include the message:  
        **"Please ask questions related to the Vendor Management System."**

        **Support Inquiries:**  
        For any support issues, refer users to **support@ultimatesolutions.com**.

        Always maintain **professionalism** and stay **focused** on **Vendor Management System-related topics**. Never attempt to answer questions outside the VMS scope.
    """
)

        history.append(system_message)

        try:
           
           
            print(request.message)
            guard.validate(request.message)

            # Convert the request history to LangChain message format
            history.extend(
                HumanMessage(content=msg["content"]) if msg["role"] == "human" else AIMessage(content=msg["content"])
                for msg in request.history
            )


            # Append the new user message
            history.append(HumanMessage(content=request.message))
            


            # Get the AI response using the agent_executor
            response = agent.invoke({"input": request.message, "chat_history": history})
            response = response.get('output', 'Sorry, I couldn\'t process your request.')
            # Debugging: Print agent response
            print("Agent response:", response)
            guard.validate(response)

            

        except Exception as e:
            print(e)
            response = "Sorry I can't help with that"     

        # Check if the agent's response contains 'output'
        ai_response = {"role": "ai", "content": response}

        # Return the response and updated history
        return {"reply": ai_response, "updated_history": request.history + [ai_response]}

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
