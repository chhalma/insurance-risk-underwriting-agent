from fastapi import FastAPI

from api.routes.policy_routes import policy_router
from api.routes.recommendation_routes import recommendation_router
from api.routes.risk_routes import risk_router

app = FastAPI(title="Insurance Underwriting Assistant")
app.include_router(risk_router)
app.include_router(policy_router)
app.include_router(recommendation_router)
