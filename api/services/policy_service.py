from functools import cached_property

from api.models.policy_schemas import PolicyAnswerResponse, PolicyQueryResponse, PolicySection
from rag_pipeline.rag_chain import PolicyAnswerChain, PolicyRetriever


class PolicyService:
    """Business logic: turns a natural-language query into relevant policy sections."""

    def __init__(self, retriever: PolicyRetriever):
        self.retriever = retriever

    def search(self, query: str, k: int) -> PolicyQueryResponse:
        results = self.retriever.retrieve(query, k=k)
        return PolicyQueryResponse(results=[PolicySection(**result) for result in results])

    @cached_property
    def _answer_chain(self) -> PolicyAnswerChain:
        # Built lazily, on first use, so the app still starts without Azure OpenAI configured.
        return PolicyAnswerChain(self.retriever)

    def answer(self, query: str, k: int) -> PolicyAnswerResponse:
        result = self._answer_chain.answer(query, k=k)
        return PolicyAnswerResponse(**result)
