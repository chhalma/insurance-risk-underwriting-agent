from fastapi import FastAPI

from api.interfaces.risk_routes import router

app = FastAPI(title="Insurance Underwriting Assistant")
app.include_router(router)
