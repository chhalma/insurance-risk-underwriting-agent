# Insurance Underwriting Assistant

An AI-powered insurance underwriting agent combining machine learning risk prediction, SHAP explainability, and Retrieval-Augmented Generation (RAG) to provide transparent health insurance risk assessments.

## Architecture

```
                 User
                  |
                  v
        Azure AI Foundry Agent
                  |
        --------------------
        |                  |
        v                  v
 Insurance RAG        Risk Prediction API
        |                  |
 Azure AI Search       XGBoost Model
        |                  |
 Policy PDFs          SHAP Explanation
        |
        v
 Explainable Insurance Recommendation
```

- **ML Model**: XGBoost trained on Kaggle medical cost dataset → predicts risk score, explained via SHAP
- **RAG Pipeline**: LangChain + FAISS over BUPA health policy → retrieves relevant policy sections
- **API**: FastAPI server → serves predictions and policy explanations

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the API: `uvicorn api.main:app --reload`

## Project Structure

- `ml-model/`: Trained model and preprocessing
- `rag-pipeline/`: Document retrieval and RAG logic
- `api/`: FastAPI endpoints for risk scoring

## Next Steps

- [ ] Add Azure AI Search integration
- [ ] Deploy to Azure Functions
- [ ] Connect to Foundry agent
- [ ] Add Application Insights monitoring
