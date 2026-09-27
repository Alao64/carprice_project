import os
import sys
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join("notebook/car_price.csv"), na_values=['-', ' ', ''])

            # Clean Estimated_Mileage column
            df['Estimated_Mileage'] = df['Estimated_Mileage'].astype(str).str.replace('km', '', regex=False).str.strip()
            df['Estimated_Mileage'] = pd.to_numeric(df['Estimated_Mileage'], errors='coerce')

            # Fix typos and inconsistencies
            df['Fuel_Type'] = df['Fuel_Type'].replace({'Petroll': 'Petrol'})
            df['Transmission'] = df['Transmission'].replace({'AutoNamatic': 'Automatic'})
            df['Condition'] = df['Condition'].replace({'Exc': 'Excellent', 'Excellent!': 'Excellent'})
            df['Color'] = df['Color'].replace({'Brwn': 'Brown'})
            df['Cruise'] = df['Cruise'].replace({'Y': 'Yes'})
            df['Insurance'] = df['Insurance'].replace({'3rd Party': 'Third Party', 'No': 'No insurance'})
            df['Service_History'] = df['Service_History'].replace({'Full ': 'Full Service', 'Partial ': 'Partial Service'})
            df['Safety'] = df['Safety'].replace({'4': '4 stars'})
            df['TAge'] = df['TAge'].replace({'4 years': 4, '50': 5})
            df['Cylinder_Numbers'] = df['Cylinder_Numbers'].replace({'3': 'three', '4': 'four', '5': 'five'})
            df['Owners'] = df['Owners'].replace({44:4,22:2})
            # Categorize Credit_History
            df['Credit_History'] = pd.to_numeric(df['Credit_History'],errors='coerce')
            df['TAge'] = pd.to_numeric(df['TAge'],errors='coerce')
            def categorize_credit(x):
             if x < -0.01:
                 return "Poor"
             elif x <= 0.01:
              return "Fair"
             else:
              return "Good"
            def doors_num(y):
              if y >= 20:
                return y/10
              else:
                return y
            def engine_size(z):
                if z >= 100:
                    return z/100
                else:
                    return z
            df['DoorsNum'] = df['DoorsNum'].apply(doors_num)
            df['Engine_Size'] = df['Engine_Size'].apply(engine_size)
            df['Credit_History'] = df['Credit_History'].apply(categorize_credit)
            #Feature Engineering
            df['Car_Age']= 2025-df['Year']
            df['Model_Brand'] = df['Model'].str.lower().str.split().str[0]
            df['Model_Type'] = df['Model'].str.lower().str.split().str[1]
            df.drop(columns=['Model','Year'], inplace=True)
            logger.info("Data cleaning completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    voting_r2, voting_rmse, tuned_scores = model_trainer.initiate_model_trainer(train_array, test_array)

    print(f"\nVotingRegressor R2 Score: {voting_r2:.4f}")
    print(f"VotingRegressor RMSE: {voting_rmse:,.2f}")

    print("\nTop 3 tuned models used in the vote:")
    for i, (name, r2, rmse) in enumerate(tuned_scores, 1):
        print(f" #{i} {name}: R2 = {r2:.4f} | RMSE = {rmse:,.2f}")