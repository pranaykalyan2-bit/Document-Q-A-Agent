import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from rag_pipeline import extract_pdf_pages, split_into_chunks, search_chunks, generate_answer

load_dotenv()

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Q&A Assistant")
st.write("Upload a PDF and ask questions about its contents. Answers are generated using RAG and include source-page references.")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    st.warning("Add GEMINI_API_KEY to your local .env file or Streamlit app Secrets before asking questions.")
    st.stop()

# The existing pipeline reads the key from the environment.
os.environ["GEMINI_API_KEY"] = api_key

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        with st.spinner("Reading PDF and preparing the search index. The first run may take a little while to download the embedding model..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_bytes)
                temp_pdf_path = temp_pdf.name

            try:
                pages = extract_pdf_pages(temp_pdf_path)
                chunks = split_into_chunks(pages)

                if not chunks:
                    st.error("I couldn't extract readable text from this PDF. If it's a scanned PDF, OCR may be required.")
                    st.stop()

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )

                # Use an in-memory Chroma collection so each uploaded PDF gets a fresh index.
                documents = [
                    Document(
                        page_content=chunk["text"],
                        metadata={"page": chunk["page"]},
                    )
                    for chunk in chunks
                ]
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    collection_name=f"pdf_{file_hash[:16]}",
                )
            finally:
                Path(temp_pdf_path).unlink(missing_ok=True)

        st.session_state["file_hash"] = file_hash
        st.session_state["pages"] = pages
        st.session_state["vector_store"] = vector_store
        st.session_state["file_name"] = uploaded_pdf.name
        st.session_state["chat_history"] = []

    st.success(f"Ready: **{st.session_state['file_name']}** · {len(st.session_state['pages'])} pages")

    with st.expander("View PDF text extraction details"):
        st.write(f"Pages extracted: {len(st.session_state['pages'])}")
        st.write(f"Text chunks indexed: {len(chunks) if 'chunks' in locals() else 'Index ready'}")

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**PDF page {source['page']}**")
                        st.write(source["text"])

    question = st.chat_input("Ask a question about this PDF...")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                try:
                    results = search_chunks(st.session_state["vector_store"], question, k=8)
                    answer = generate_answer(question, results)
                    sources = [
                        {
                            "page": result.metadata.get("page", "?"),
                            "text": result.page_content,
                        }
                        for result in results
                    ]
                    st.markdown(answer)
                    with st.expander("Sources used"):
                        for source in sources:
                            st.markdown(f"**PDF page {source['page']}**")
                            st.write(source["text"])

                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")
else:
    st.info("Upload a PDF above to get started.")
