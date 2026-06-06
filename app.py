import os
import streamlit as st
import chromadb
import subprocess
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# SECRETS MANAGEMENT (Works locally + Streamlit Cloud)
# =========================================================
OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY") 
    or os.getenv("OPENROUTER_API_KEY")
)

if not OPENROUTER_API_KEY:
    st.error("❌ OPENROUTER_API_KEY is missing!")
    st.info("""
    **How to fix:**
    - Local: Add it to your `.env` file
    - Streamlit Cloud: Go to Settings → Secrets and add it
    """)
    st.stop()

# =========================================================
# AUTO INGEST - Build ChromaDB if not exists
# =========================================================
@st.cache_resource(show_spinner=False)
def init_chromadb():
    chroma_path = "./chroma_db"
    
    # Run ingest.py if database doesn't exist or is empty
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        st.info("🔨 First time setup: Building vector database...")
        try:
            result = subprocess.run(
                ["python", "ingest.py"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            st.success("✅ Vector database created successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Failed to run ingest.py:\n{e.stderr}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error during ingest: {e}")
            st.stop()

    # Initialize client and collection
    client = chromadb.PersistentClient(path=chroma_path)
    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_collection(
        name="company_docs",
        embedding_function=embedding_func
    )
    return collection

# =========================================================
# LLM INITIALIZATION
# =========================================================
@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=600
    )

# =========================================================
# MAIN APP
# =========================================================
try:
    collection = init_chromadb()
    llm = init_llm()

    # Title and Header
    st.title("🤖 Company Knowledge Assistant")
    st.markdown("Ask me anything about **company policies**!")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant can answer questions about:
        - Vacation & Time Off
        - Remote Work Policy
        - Parental Leave
        - Benefits
        - IT Policies
        """)
        st.divider()
        st.metric("Documents Indexed", collection.count())
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Session State for Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome Message
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.write("""
            Hi! 👋 I'm your **Company Knowledge Assistant**. 
            
            I can help you with HR policies, benefits, and more.
            Just type your question below!
            """)

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # RAG Response Function
    def get_rag_response(question, n_results=3):
        try:
            results = collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            contexts = results.get("documents", [[]])[0]
            
            if not contexts:
                return "Sorry, I couldn't find any relevant information in the company documents."

            context_text = "\n\n".join(contexts)

            system_prompt = """
You are a helpful and accurate company knowledge assistant.
- Answer ONLY using the provided context.
- If the answer is not in the context, say: "I don't have information about that."
- Do not make up any information.
- Keep answers clear and professional.
"""

            user_prompt = f"""
Context:
{context_text}

Question:
{question}
"""

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content

        except Exception as e:
            return f"⚠️ Error processing your question: {str(e)}"

    # Chat Input
    if prompt := st.chat_input("Ask a question about company policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching company documents..."):
                response = get_rag_response(prompt)
                st.write(response)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"Application Error: {str(e)}")
    st.info("Please make sure `ingest.py` and your documents are in the correct location.")
