import pytest
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.rag_chain import EMBEDDING_MODEL, build_index, load_policy_documents


@pytest.fixture(scope="session")
def embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def test_load_policy_documents_raises_when_no_pdfs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy_documents(tmp_path)


def test_build_index_and_retrieve_finds_relevant_chunk(embeddings):
    documents = [
        Document(page_content="Hospital stays are covered up to 30 days per year under the standard plan."),
        Document(page_content="Dental checkups are covered twice a year under the standard plan."),
        Document(page_content="Travel insurance claims must be filed within 14 days of the incident."),
    ]

    vector_store = build_index(documents, embeddings)
    results = vector_store.similarity_search("How many days of hospital stay are covered?", k=1)

    assert "Hospital stays" in results[0].page_content
