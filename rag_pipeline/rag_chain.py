import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RAG_ROOT = Path(__file__).resolve().parent
POLICY_DOCUMENTS_DIR = RAG_ROOT / "policy_documents"
INDEX_PATH = RAG_ROOT / "vector_store" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_policy_documents(policy_documents_dir: Path = POLICY_DOCUMENTS_DIR) -> list:
    """Loads every PDF in the policy documents folder into LangChain Documents."""
    pdf_paths = sorted(policy_documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {policy_documents_dir}. Add a policy document before building the index."
        )

    documents = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return documents


def build_index(documents: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Chunks documents and embeds them into a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


class PolicyRetriever:
    """Retrieves relevant policy document sections to ground underwriting explanations."""

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


class PolicyAnswerChain:
    """Combines retrieval with LLM generation to produce a grounded, cited answer."""

    def __init__(self, retriever: PolicyRetriever):
        self.retriever = retriever
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    def answer(self, query: str, k: int = 3) -> dict:
        sections = self.retriever.retrieve(query, k=k)
        context = "\n\n".join(
            f"[Source: page {section['page']}]\n{section['content']}" for section in sections
        )

        prompt = (
            "You are an insurance policy assistant. Answer the question using ONLY the "
            "policy excerpts below. If the excerpts don't contain the answer, say so — "
            "don't guess.\n\n"
            f"Policy excerpts:\n{context}\n\n"
            f"Question: {query}"
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": sections,
        }


if __name__ == "__main__":
    retriever = PolicyRetriever()
    for result in retriever.retrieve("What is covered for hospital stays?"):
        print(result)
