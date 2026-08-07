from pathlib import Path

import joblib
import shap
import yaml
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class RiskModelRepository:
    """Owns access to the trained model artifacts (model, preprocessor, explainer)."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.model = XGBRegressor()
        self.model.load_model(PROJECT_ROOT / config["model"]["path"])
        self.preprocessor = joblib.load(PROJECT_ROOT / config["model"]["preprocessor_path"])
        self.feature_names = self.preprocessor.get_feature_names_out()
        self.explainer = shap.TreeExplainer(self.model)

        thresholds = config["risk_thresholds"]
        self.low_max = thresholds["low_max"]
        self.medium_max = thresholds["medium_max"]
