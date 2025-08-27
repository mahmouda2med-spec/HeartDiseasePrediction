from pydantic import BaseModel

# Model for request (input features)
class HEARTPredictionRequest(BaseModel):
    age: int
    sex: int
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalachh: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
    # الأعمدة الجديدة اللي أضفتها
    heart_stress_index: float
    vessel_severity_score: float

# Model for response (prediction result)
class HEARTPredictionResponse(BaseModel):
    target: int
