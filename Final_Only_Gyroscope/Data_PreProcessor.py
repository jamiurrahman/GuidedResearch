import pandas as pd

class Data_PreProcessor:
    def get_data_for_all_sensors (self):
        csv_path = "./RawData/dataset_all_sensors.csv"

        df = pd.read_csv(csv_path)

        return df
