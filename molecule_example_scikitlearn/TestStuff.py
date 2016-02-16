from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


def random_forest(classifier):
    # create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('Data/train.csv', delimiter=',')[1:]
    train, test = train_test_split(dataset, train_size = 0.8)

    train_target = [x[0] for x in train]
    train_data = [x[1:] for x in train]

    test_target = [x[0] for x in test]
    test_data = [x[1:] for x in test]
    
    rf = classifier

    #Fit using training data
    rf.fit(train_data, train_target)

    #Predict using the hold back set
    prediction = rf.predict_proba(test_data)
    index = 0;
    true_positive=0
    true_negative=0
    false_positive=0
    false_negative=0
    for pred in prediction:
        if test_target[index] == 0:
            if pred[0] >= 0.5:
                true_negative += 1
            else:
                false_positive += 1

        if test_target[index] == 1:
            if pred[1] > 0.5:
                true_positive += 1
            else:
                false_negative += 1

        index += 1

    print("True Positive: " + str(true_positive))
    print("True Negative: " + str(true_negative))
    print("False Positive: " + str(false_positive))
    print("False Negative: " + str(false_negative))

    print("Overall \"correctness\":" + str((float(true_positive + true_negative) /
          (true_positive+true_negative+false_positive+false_negative))))

    print("Total data elements %d, of which %d are wrongly classified" % (len(test_data), false_negative + false_positive))
    #predicted_probs = [[index + 1, x[1]] for index, x in enumerate(prediction)]


    #savetxt('Data/submission_500.csv', predicted_probs, delimiter=',', fmt='%d,%f',
    #             header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    print ("Random forest")
    random_forest(RandomForestClassifier(n_estimators=100))
    print("Naive Bayes - Gaussian")
    random_forest(GaussianNB())
    print("Naive Bayes - Bernoullian")
    random_forest(BernoulliNB())
    print("Naive Bayes - Multinomial")
    random_forest(MultinomialNB())