import pytest

from api.repository.risk_model_repository import RiskModelRepository
from api.services.risk_service import RiskService


@pytest.fixture(scope="session")
def repository():
    return RiskModelRepository()


@pytest.fixture(scope="session")
def risk_service(repository):
    return RiskService(repository)
