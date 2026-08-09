import os
from typing import Protocol

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class PolicySectionRetriever(Protocol):
    """Anything that can look up relevant policy sections for a query — local FAISS or Azure AI Search."""

    def retrieve(self, query: str, k: int = 3) -> list[dict]: ...


class PolicyAnswerChain:
    """Combines retrieval with LLM generation to produce a grounded, cited answer."""

    def __init__(self, retriever: PolicySectionRetriever):
        self.retriever = retriever
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

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

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sections,
        }
