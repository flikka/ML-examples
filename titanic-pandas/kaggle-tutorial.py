import pandas as pd
import pylab as P
import numpy as np

# From https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

def read_dataframe(filename):
    """@rtype: pandas.DataFrame"""
    data_frame = pd.read_csv(filename, header=0)
    return data_frame

def main():
    df = read_dataframe('train.csv')

    df['Gender'] = df.Sex.map({'female':0, 'male':1}).astype(int)

    # Going to create replacements for NaN ages, set to median of the
    # given gender (0=female, 1=male) in the three passenger classes (1,2,3)
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & \
                                  (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']
    for gender_idx in range(0, 2):
        for p_class_idx in range(0, 3):
            df.loc[(df.Gender == gender_idx) & (df.Pclass == p_class_idx +1) \
                & df.Age.isnull(), 'AgeFill'] = median_ages[gender_idx, p_class_idx]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass


    df_surived_AgeClass = df[df['Survived'] == 1]['Age']
    df_died_AgeClass = df[df['Survived'] == 0]['Age']

    print(df_surived_AgeClass)

    bins=np.arange(min(df['Age']), max(df['Age']) + 10, 10)

    P.hist(df_surived_AgeClass, bins, alpha=0.5, label='survived')
    P.hist(df_died_AgeClass, bins, alpha=0.5, label='died')
    P.legend(loc='upper right')
    P.show()

if __name__ == "__main__":
    main()

