import pandas as pd
import pylab as P

def read_dataframe(filename):
    """@rtype: pandas.DataFrame"""
    data_frame = pd.read_csv(filename, header=0)
    return data_frame

def main():
    df = read_dataframe('train.csv')
    #print(df[(df.Sex == 'male') & (df.Age > 60)].Cabin)

    #df.Age.hist(bins=36)
    #P.show()

    df['Gender'] = df.Sex.map({'female':0, 'male':1}).astype(int)
    

if __name__ == "__main__":
    main()

