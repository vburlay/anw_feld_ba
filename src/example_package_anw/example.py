import pandas as pd

def source_date(path_to_file) :
    data = pd.read_csv(path_to_file)
    return data
