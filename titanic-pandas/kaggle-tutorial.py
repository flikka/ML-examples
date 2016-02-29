import pandas as pd
import pylab as P
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# From https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii

def read_dataframe(filename):
    """@rtype: pandas.DataFrame"""
    data_frame = pd.read_csv(filename, header=0)
    return data_frame

def main():
    df_train = read_dataframe('train.csv')
    df_test = read_dataframe('test.csv')
    processed_train = clean_and_expand_titanic(df_train)
    processed_train = processed_train.drop('PassengerId', axis=1)
    processed_train = processed_train.drop('Name', axis=1)
    processed_train = processed_train.drop('Age', axis=1)
    processed_train = processed_train.drop('Ticket', axis=1)
    processed_train = processed_train.drop('Cabin', axis=1)
    processed_train = processed_train.drop('Embarked', axis=1)
    
    processed_test = clean_and_expand_titanic(df_test)
    processed_test = processed_test.drop('PassengerId', axis=1)
    processed_test = processed_test.drop('Name', axis=1)
    processed_test = processed_test.drop('Age', axis=1)
    processed_test = processed_test.drop('Ticket', axis=1)
    processed_test = processed_test.drop('Cabin', axis=1)
    processed_test = processed_test.drop('Embarked', axis=1)

    print("\nTraining set, as data frame")
    print(processed_train.describe())

    print("\nTesting set, as data frame")
    print(processed_test.describe())

    assert(not df_train.equals(processed_train))
    assert(not df_test.equals(processed_test))

    train_data = processed_train.values
    test_data = processed_test.values

    predict_with_random_forest(train_data, test_data)

def clean_and_expand_titanic(input_data_frame):
    df = input_data_frame.copy()
    df['Gender'] = df.Sex.map({'female': 0, 'male': 1}).astype(int)

    
    df['EmbarkedAsInt'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
    # Going to create replacements for NaN ages, set to median of the
    # given gender (0=female, 1=male) in the three passenger classes (1,2,3)
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = df[(df['Gender'] == i) & \
                                   (df['Pclass'] == j + 1)]['Age'].dropna().median()
    df['AgeFill'] = df['Age']
    for gender_idx in range(0, 2):
        for p_class_idx in range(0, 3):
            df.loc[(df.Gender == gender_idx) & (df.Pclass == p_class_idx + 1) \
                   & df.Age.isnull(), 'AgeFill'] = median_ages[gender_idx, p_class_idx]
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass
    return df

def predict_with_random_forest(training_data, test_data):
    # Create the random forest object which will include all the parameters
    # for the fit
    forest = RandomForestClassifier(n_estimators = 100)
    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(training_data[0::,1::],training_data[0::,0])


if __name__ == "__main__":
    main()

