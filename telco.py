from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

models = joblib.load("telco_churn_all_models.pkl")


class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(
    directory="templates"
)

from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict")
def predict(data: Customer):

    try:
        df = pd.DataFrame([data.dict()])

        print("Input DF:")
        print(df)

        results = []

        for model_name, model in models.items():

            print("Running:", model_name)

            pred = model.predict(df)[0]

            results.append({
                "Model": model_name,
                "Prediction": int(pred)
            })

        return {"predictions": results}

    except Exception as e:
        print("ERROR OCCURRED:")
        print(str(e))
        return {"error": str(e)}