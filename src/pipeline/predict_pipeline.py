import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

    def explain(self, features):
        """SHAP values for one row, using the best individual tree model."""
        try:
            model_path = os.path.join("artifacts", "best_individual_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_scaled)

            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]
            base_value = float(base_value)

            return shap_values[0], feature_names, base_value

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        DoorsNum: int,
        Owners: int,
        Warranty: int,
        Engine_Size: float,
        Weight: float,
        carlength: float,
        carwidth: float,
        monthly_mileage: float,
        peakrpm: float,
        Estimated_Mileage: float,
        Car_Age: int,
        Model_Brand: str,
        Model_Type: str,
        Fuel_Type: str,
        Transmission: str,
        Condition: str,
        Color: str,
        Cruise: str,
        Leather_Seats: str,
        Heated_Seats: str,
        Navigation: str,
        Insurance: str,
        Service_History: str,
        Safety: str,
        Premium_Sound: str,
        Multimedia: str,
        Bluetooth: str,
        Wheel: str,
        Sunroof: str,
        TAge: str,
        Cylinder_Numbers: str,
        Credit_History: str,
    ):
        self.DoorsNum = DoorsNum
        self.Owners = Owners
        self.Warranty = Warranty
        self.Engine_Size = Engine_Size
        self.Weight = Weight
        self.carlength = carlength
        self.carwidth = carwidth
        self.monthly_mileage = monthly_mileage
        self.peakrpm = peakrpm
        self.Estimated_Mileage = Estimated_Mileage
        self.Car_Age = Car_Age
        self.Model_Brand = Model_Brand
        self.Model_Type = Model_Type
        self.Fuel_Type = Fuel_Type
        self.Transmission = Transmission
        self.Condition = Condition
        self.Color = Color
        self.Cruise = Cruise
        self.Leather_Seats = Leather_Seats
        self.Heated_Seats = Heated_Seats
        self.Navigation = Navigation
        self.Insurance = Insurance
        self.Service_History = Service_History
        self.Safety = Safety
        self.Premium_Sound = Premium_Sound
        self.Multimedia = Multimedia
        self.Bluetooth = Bluetooth
        self.Wheel = Wheel
        self.Sunroof = Sunroof
        self.TAge = TAge
        self.Cylinder_Numbers = Cylinder_Numbers
        self.Credit_History = Credit_History

    def get_data_as_dataframe(self):
        try:
            data = {
                "DoorsNum": [self.DoorsNum],
                "Owners": [self.Owners],
                "Warranty": [self.Warranty],
                "Engine_Size": [self.Engine_Size],
                "Weight": [self.Weight],
                "carlength": [self.carlength],
                "carwidth": [self.carwidth],
                "monthly_mileage": [self.monthly_mileage],
                "peakrpm": [self.peakrpm],
                "Estimated_Mileage": [self.Estimated_Mileage],
                "Car_Age": [self.Car_Age],
                "Model_Brand": [self.Model_Brand],
                "Model_Type": [self.Model_Type],
                "Fuel_Type": [self.Fuel_Type],
                "Transmission": [self.Transmission],
                "Condition": [self.Condition],
                "Color": [self.Color],
                "Cruise": [self.Cruise],
                "Leather_Seats": [self.Leather_Seats],
                "Heated_Seats": [self.Heated_Seats],
                "Navigation": [self.Navigation],
                "Insurance": [self.Insurance],
                "Service_History": [self.Service_History],
                "Safety": [self.Safety],
                "Premium_Sound": [self.Premium_Sound],
                "Multimedia": [self.Multimedia],
                "Bluetooth": [self.Bluetooth],
                "Wheel": [self.Wheel],
                "Sunroof": [self.Sunroof],
                "TAge": [self.TAge],
                "Cylinder_Numbers": [self.Cylinder_Numbers],
                "Credit_History": [self.Credit_History],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)