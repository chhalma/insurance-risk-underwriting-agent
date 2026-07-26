from fastapi import APIRouter

from api.models.schemas import ApplicantRequest, PredictionResponse
from api.repository.model_repository import ModelRepository
from api.services.risk_service import RiskService

router = APIRouter()
_repository = ModelRepository()
_service = RiskService(_repository)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantRequest):
    return _service.score(applicant)
