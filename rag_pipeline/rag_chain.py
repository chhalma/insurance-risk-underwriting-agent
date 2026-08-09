from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.document_loader import POLICY_DOCUMENTS_DIR, RAG_ROOT, load_policy_documents

INDEX_PATH = RAG_ROOT / "vector_store" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(documents: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Chunks documents and embeds them into a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


class PolicyRetriever:
    """Retrieves relevant policy document sections to ground underwriting explanations.

    Local FAISS prototype — free, runs entirely on-device. See azure_search_retriever.py
    for the cloud-backed equivalent used by the deployed API.
    """

    def __init__(
        self,
        policy_documents_dir: Path = POLICY_DOCUMENTS_DIR,
        index_path: Path = INDEX_PATH,
    ):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        if index_path.exists():
            self.vector_store = FAISS.load_local(
                str(index_path), self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            documents = load_policy_documents(policy_documents_dir)
            self.vector_store = build_index(documents, self.embeddings)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(index_path))

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
            for doc in results
        ]


if __name__ == "__main__":
    retriever = PolicyRetriever()
    for result in retriever.retrieve("What is covered for hospital stays?"):
        print(result)
