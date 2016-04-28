import pandas
dataset = pandas.read_csv('Data/train.csv', delimiter=',')
print(dataset.describe())