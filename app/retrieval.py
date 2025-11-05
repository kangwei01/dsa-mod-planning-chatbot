from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# where the vectors are stored --> faster loading during runtime
vector_path = Path(__file__).resolve().parent / "curriculum_info_vectors"
# what model is used
model = "mxbai-embed-large"

# build helper to ask local ollama model to turn text into embeddings
# lru_cache: to avoid re-initializing the model multiple times
@lru_cache(maxsize=1)
def embedding():
    return OllamaEmbeddings(model=model)

# returns the path to vectors
def get_vectors_path():
    return vector_path

# internal function --> load vector store from disk
@lru_cache(maxsize=1)
def vectorstore():
    # load the embedding model
    embedding_model = embedding()
    # load the vector store from disk
    # allow_dangerous_deserialization=True: to allow loading custom objects --> needed for ollamaembeddings
    return FAISS.load_local(str(vector_path), embedding_model, allow_dangerous_deserialization=True)

# expose retriever
@lru_cache(maxsize=1)
def get_retriever():
    # get the vector store
    store = vectorstore()
    # return a retriever with similarity search
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# format documents for display
def format_documents(documents, max_chars=500):
    if not documents:
        return ""

    snippets = []
    # for each document
    for index, doc in enumerate(documents, start=1):
        # get the text content
        text = doc.page_content.strip()
        # if there is a max_chars set
        if max_chars is not None and max_chars > 0:
            # truncate at the max_chars
            text = text[: max_chars].rstrip()
        # add to snippets with index
        snippets.append(f"Doc {index}: {text}")
    # join snippets with double newlines --> better readability
    return "\n\n".join(snippets)

# combines all the document contents into a single context string
def combine_context(documents):
    if not documents:
        return ""

    # get the text parts from each document, stripping whitespace
    parts = (doc.page_content.strip() for doc in documents)
    # join non-empty parts with double newlines
    return "\n\n".join(part for part in parts if part)