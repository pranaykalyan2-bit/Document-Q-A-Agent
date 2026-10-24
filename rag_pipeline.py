

from langchain_huggingface import HuggingFaceEmbeddings
import pymupdf # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract text from each PDF page and preserve page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No readable text found. This PDF may be scanned."
        )

    return pages


def split_into_chunks(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """Split page text into overlapping chunks with page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract text from each PDF page and preserve page numbers."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No readable text found. This PDF may be scanned."
        )

    return pages


def split_into_chunks(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> list[dict]:
    """Split page text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


def create_vector_store(chunks: list[dict]) -> Chroma:
    """Create local embeddings and store chunks in ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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
        persist_directory="./chroma_db",
        collection_name="pdf_documents",
    )

    return vector_store

def search_chunks(
    vector_store: Chroma,
    question: str,
    k: int = 8,
) -> list[Document]:
    """Retrieve relevant and diverse passages from the PDF."""

    results = vector_store.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=30,
        lambda_mult=0.5,
    )

    return results

import os
from langchain_google_genai import ChatGoogleGenerativeAI


def generate_answer(
    question: str,
    retrieved_docs: list[Document],
) -> str:
    """Generate an answer grounded in retrieved PDF passages."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    context = "\n\n".join(
        f"[PDF page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0,
    )


    prompt = f"""
You are a document question-answering assistant.

Answer the question using the supplied PDF excerpts.

Instructions:
- Answer every part of the question.
- If asked for a list, search the supplied excerpts for all items.
- Use only information supported by the excerpts.
- Cite page numbers as [Page 4].
- If information is missing, clearly say which details are missing.
- Do not claim the entire document lacks information just because
  it is absent from the retrieved excerpts.

PDF excerpts:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return str(content)

def find_virtualization_levels(pages: list[dict]) -> list[dict]:
    """Find the five numbered virtualization-level sections."""

    import re

    levels = []

    for page in pages:
        text = page["text"]

        matches = list(re.finditer(
            r"3\.1\.1\.[1-5]\s+([^\n]+)",
            text,
            flags=re.IGNORECASE,
        ))

        for match in matches:
            levels.append({
                "section": match.group(0).strip(),
                "page": page["page"],
                "context": text[match.start():match.start() + 1200],
            })

    return levels
