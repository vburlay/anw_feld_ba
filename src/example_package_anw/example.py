import pandas as pd
import os


def source_date(path):
    path_to_file = os.getcwd() + path
    data = pd.read_csv(path_to_file)
    return data
