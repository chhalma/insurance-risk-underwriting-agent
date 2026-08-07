from pydantic import BaseModel


class PolicyQueryRequest(BaseModel):
    query: str
    k: int = 3


class PolicySection(BaseModel):
    content: str
    source: str
    page: int | None = None


class PolicyQueryResponse(BaseModel):
    results: list[PolicySection]


class PolicyAnswerResponse(BaseModel):
    answer: str
    sources: list[PolicySection]
