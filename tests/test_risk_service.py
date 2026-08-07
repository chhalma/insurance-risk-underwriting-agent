from api.models.risk_model_schemas import ApplicantRequest
from api.services.risk_service import RiskService


class _StubRepository:
    """Minimal repository double for testing categorization logic without loading artifacts."""

    def __init__(self, low_max: float, medium_max: float):
        self.low_max = low_max
        self.medium_max = medium_max


class TestCategorize:
    def _service(self):
        return RiskService(_StubRepository(low_max=5000, medium_max=15000))

    def test_below_low_max_is_low(self):
        assert self._service()._categorize(4999) == "Low"

    def test_at_low_max_is_medium(self):
        assert self._service()._categorize(5000) == "Medium"

    def test_between_thresholds_is_medium(self):
        assert self._service()._categorize(10000) == "Medium"

    def test_at_or_above_medium_max_is_high(self):
        assert self._service()._categorize(15000) == "High"
        assert self._service()._categorize(50000) == "High"


class TestScoreIntegration:
    """Exercises the real trained model + preprocessor end to end."""

    def _applicant(self, **overrides):
        defaults = dict(age=30, sex="male", bmi=28, children=0, smoker="no", region="northeast")
        defaults.update(overrides)
        return ApplicantRequest(**defaults)

    def test_returns_well_formed_prediction(self, risk_service):
        result = risk_service.score(self._applicant())

        assert result.predicted_annual_charge > 0
        assert result.risk_category in {"Low", "Medium", "High"}
        assert len(result.key_risk_factors) == 3

    def test_smoker_predicted_higher_than_non_smoker(self, risk_service):
        smoker = risk_service.score(self._applicant(smoker="yes"))
        non_smoker = risk_service.score(self._applicant(smoker="no"))

        assert smoker.predicted_annual_charge > non_smoker.predicted_annual_charge

    def test_older_applicant_predicted_higher_than_younger(self, risk_service):
        older = risk_service.score(self._applicant(age=60))
        younger = risk_service.score(self._applicant(age=20))

        assert older.predicted_annual_charge > younger.predicted_annual_charge

    def test_smoking_is_the_top_risk_factor_for_smokers(self, risk_service):
        result = risk_service.score(self._applicant(smoker="yes"))

        assert "smoker" in result.key_risk_factors[0].feature
