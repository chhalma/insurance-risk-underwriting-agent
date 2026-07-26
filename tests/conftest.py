import pytest

from api.repository.model_repository import ModelRepository
from api.services.risk_service import RiskService


@pytest.fixture(scope="session")
def repository():
    return ModelRepository()


@pytest.fixture(scope="session")
def risk_service(repository):
    return RiskService(repository)
