import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Example preprocessing steps
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp')
    return data
