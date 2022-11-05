from typing import Optional
from pandas import read_csv

class DataSource():
    data: list[float]
    outputLabels: Optional[list[float]]

    neighborMatrix: list[int]

    def __init__(self, data, outputLabels):
        self.data = data
        self.outputLabels = outputLabels

    def number_atributes(self):
        return len(self.data[0])

    def number_data_points(self):
        return len(self.data)


def read_data_file(filename: str, hasLabels: bool=True, header=None, na_values: list[str]= []) -> DataSource:

    df = read_csv(filename, header=header, na_values=na_values)
    
    if hasLabels:
        dataPoints = df.iloc[:, 0:len(df.columns)-1].values.tolist()
        outputLabels = df.iloc[:, len(df.columns)-1:len(df.columns)].values.tolist()
    else:
        dataPoints = df.values.tolist()
        outputLabels = []

    data = DataSource(data=dataPoints, outputLabels=outputLabels)

    return data
