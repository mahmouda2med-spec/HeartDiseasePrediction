import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from domain.domain import HEARTPredictionRequest, HEARTPredictionResponse
class HeartPredictionService:
    def __init__(self):
        # مسار الموديل المدرب
        self.path_model = os.path.join(os.path.dirname(__file__), "..", "artifacts", "randomforest_model.pkl")
        self.path_model = os.path.abspath(self.path_model)

        # تحميل الموديل
        self.model = self.load_artifact(self.path_model)

    def load_artifact(self, path_to_artifact):
        print("Current working directory:", os.getcwd())
        print("Trying to load model from:", path_to_artifact)
        with open(path_to_artifact, 'rb') as f:
            artifact = pickle.load(f)
        return artifact

    def preprocess_input(self, request: HEARTPredictionRequest) -> pd.DataFrame:
        """تجهيز الداتا قبل تدخل الموديل"""
        data_dict = {
            "age": request.age,
            "sex": request.sex,
            "cp": request.cp,
            "trestbps": request.trestbps,
            "chol": request.chol,
            "fbs": request.fbs,
            "restecg": request.restecg,
            "thalachh": request.thalachh,
            "exang": request.exang,
            "oldpeak": request.oldpeak,
            "slope": request.slope,
            "ca": request.ca,
            "thal": request.thal,
            "heart_stress_index": request.heart_stress_index,
            "vessel_severity_score": request.vessel_severity_score
        }
        data_df = pd.DataFrame.from_dict([data_dict])
        return data_df

    def predict(self, request: HEARTPredictionRequest) -> HEARTPredictionResponse:
        """تشغيل الموديل وإرجاع النتيجة"""
        input_data = self.preprocess_input(request)
        prediction = self.model.predict(input_data)[0]  # 0 = لا يوجد مرض, 1 = يوجد مرض
        return HEARTPredictionResponse(target=int(prediction))