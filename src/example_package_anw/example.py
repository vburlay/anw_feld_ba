import pandas as pd
import os

def source_date(path_to_file) :
    data = pd.read_csv(path_to_file)
    return data
