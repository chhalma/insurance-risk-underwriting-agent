from fastapi import APIRouter

from api.models.policy_schemas import PolicyAnswerResponse, PolicyQueryRequest, PolicyQueryResponse
from api.services.policy_service import PolicyService
from rag_pipeline.azure_search_retriever import AzureSearchPolicyRetriever

policy_router = APIRouter()
_retriever = AzureSearchPolicyRetriever()
_service = PolicyService(_retriever)


@policy_router.post("/policy/search", response_model=PolicyQueryResponse)
def search_policy(request: PolicyQueryRequest):
    return _service.search(request.query, k=request.k)


@policy_router.post("/policy/answer", response_model=PolicyAnswerResponse)
def answer_policy_question(request: PolicyQueryRequest):
    return _service.answer(request.query, k=request.k)
