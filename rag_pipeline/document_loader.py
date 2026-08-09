from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

RAG_ROOT = Path(__file__).resolve().parent
POLICY_DOCUMENTS_DIR = RAG_ROOT / "policy_documents"


def load_policy_documents(policy_documents_dir: Path = POLICY_DOCUMENTS_DIR) -> list[Document]:
    """Loads every PDF in the policy documents folder into LangChain Documents.

    Uses pypdf directly rather than langchain_community's PyPDFLoader wrapper — that
    wrapper's package import eagerly pulls in PyTorch (~1000 extra modules) via unrelated
    loader integrations, even though plain PDF text extraction never touches it.
    """
    pdf_paths = sorted(policy_documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {policy_documents_dir}. Add a policy document before building the index."
        )

    documents = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages):
            documents.append(
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": str(pdf_path), "page": page_number},
                )
            )
    return documents
