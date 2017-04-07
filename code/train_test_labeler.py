
from __future__ import print_function
import fileinput
from glob import glob
import sys

from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.externals import six
import preparetestdata
import nltk
import codecs
from sklearn.externals import joblib



import sequence_learner

def test_model():
    
    print("Loading test data...", end=" ")
    test_files = [f for i, f in enumerate(glob("../training data/labelled test data/resume0.txt"))]
    test = load_conll(fileinput.input(test_files), sequence_learner.features)
    X_test, _, lengths_test = test
    sequence_learner.describe(X_test, lengths_test)
    
    X_test, y_test, lengths_test = test
    
    clf = joblib.load('model/seq_labeler.pkl')
    y_pred = clf.predict(X_test, lengths_test)
    
    target = codecs.open("../training data/test data/test results/resume_res", "w", "utf-8")
    
    for i in range(0, X_test.shape[0]):
        target.write(y_pred[i]+"\n")
        
    print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
        
        
def train_model():
    train = sequence_learner.load_data()
    X_train, y_train, lengths_train = train
    
    clf = joblib.load('model/seq_labeler.pkl')
    clf.fit(X_train, y_train, lengths_train)

    joblib.dump(clf, 'model/seq_labeler.pkl') 
    
    
def main():
    #train_model()
    test_model()
    
if __name__=="__main__":
    main()
