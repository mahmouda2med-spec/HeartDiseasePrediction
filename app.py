from fastapi import FastAPI
from domain.domain import HEARTPredictionRequest, HEARTPredictionResponse
from service.HeartPrediction_servie import HeartPredictionService

# إنشاء تطبيق FastAPI
heart_app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="API للتنبؤ بأمراض القلب باستخدام بيانات المريض الطبية"
)

# إنشاء instance من الخدمة
prediction_service = HeartPredictionService()

# Endpoint للتنبؤ
@heart_app.post("/predict", response_model=HEARTPredictionResponse)
async def predict_heart_disease(request: HEARTPredictionRequest) -> HEARTPredictionResponse:
    """
    تنبؤ بإمكانية الإصابة بأمراض القلب بناءً على البيانات الطبية
    
    - **request**: بيانات المريض الطبية
    - **returns**: نتيجة التنبؤ (0 = لا يوجد مرض, 1 = يوجد مرض)
    """
    return prediction_service.predict(request=request)
