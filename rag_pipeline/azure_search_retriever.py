import hashlib
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

from rag_pipeline.document_loader import POLICY_DOCUMENTS_DIR, load_policy_documents

load_dotenv()

EMBEDDING_DIMENSIONS = 1536


def _get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def _embed_documents(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"], input=texts)
    return [item.embedding for item in response.data]


def _get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


def _get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


def create_search_index() -> None:
    """Creates the policy documents index in Azure AI Search if it doesn't already exist."""
    index_client = _get_search_index_client()
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    if index_name in [index.name for index in index_client.list_indexes()]:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="default-vector-profile",
        ),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[
            VectorSearchProfile(name="default-vector-profile", algorithm_configuration_name="default-hnsw")
        ],
    )

    index_client.create_index(SearchIndex(name=index_name, fields=fields, vector_search=vector_search))


def upload_policy_documents() -> int:
    """Chunks and embeds policy PDFs, then uploads them to Azure AI Search. Returns the number of chunks uploaded."""
    documents = load_policy_documents(POLICY_DOCUMENTS_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    client = _get_openai_client()
    vectors = _embed_documents(client, [chunk.page_content for chunk in chunks])

    search_docs = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        doc_id = hashlib.sha256(f"{source}-{page}-{i}".encode()).hexdigest()
        search_docs.append(
            {
                "id": doc_id,
                "content": chunk.page_content,
                "content_vector": vector,
                "source": source,
                "page": page,
            }
        )

    _get_search_client().upload_documents(documents=search_docs)
    return len(search_docs)


class AzureSearchPolicyRetriever:
    """Retrieves relevant policy sections from Azure AI Search using hybrid (vector + keyword) search."""

    def __init__(self):
        self.client = _get_openai_client()
        self.search_client = _get_search_client()

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vector = _embed_documents(self.client, [query])[0]
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="content_vector")

        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["content", "source", "page"],
            top=k,
        )

        return [{"content": r["content"], "source": r["source"], "page": r["page"]} for r in results]
