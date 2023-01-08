import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def predict(clf, text, vectorizer):
    # form an array that matches the features accounted for in the vectorizer
    array = np.zeros((1, len(vectorizer.get_feature_names_out())))
    counter = 0
    # fill in each feature manually
    for name in vectorizer.get_feature_names_out():
        if text.lower().count(name):
            array[0][counter] = text.lower().count(name)
        counter+= 1
    # return the prediction of the classifier
    return clf.predict(array)

if __name__ == "__main__":
    # load into dataframe
    df = pd.read_csv("dataset.csv")
    x = df["Text"]
    y = df["language"]
    print("[...] Data is loaded!")

    # fit model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(x)
    clf = MultinomialNB(alpha = 0.0001)
    clf.fit(X, y) # no reason to do a split so we can use the full dataset
    print("[...] Model is trained!")
    print("[...] Enter a string to get its language. you can exit by typing !q")
    
    instr = input("[<<<]: ")
    while not instr in ["!q", "!quit", "!exit"]:
        print("[>>>] The language seemse to be:", predict(clf, instr, vectorizer)[0])
        instr = input("[<<<]: ")
