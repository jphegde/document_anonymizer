
"""
Generic sequence prediction script using CoNLL format.
"""

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



def features(sentence, i):
    """Features for i'th token in sentence.

    Currently baseline named-entity recognition features, but these can
    easily be changed to do POS tagging or chunking.
    """

    word = sentence[i]
    
    #print(sentence)
    yield word


def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))


def load_data():
    files = glob("../training data/new training/new ent 2/*.txt")#"../training data/new training/new_file*.txt")

    # 80% training, 20% test
    print("Loading training data...", end=" ")
    """train_files = [f for i, f in enumerate(files) if i % 5 != 0]
    train = load_conll(fileinput.input(train_files), features)#, split=True)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)"""
    
    
    train_files = [f for i, f in enumerate(files)]# if i % 5 != 0]
    print( train_files)
    train = load_conll(fileinput.input(train_files), features)#, split=True)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)

    print("Loading test data...", end=" ")
  
    """test_files = [f for i, f in enumerate(glob("../training data/new training/new_test.txt"))]
    test = load_conll(fileinput.input(test_files), features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)"""

    return train#, test


if __name__ == "__main__":
    print(__doc__)

    #print("Loading training data...", end=" ")
    #X_train, y_train, lengths_train = load_conll(sys.argv[1], features)
    #describe(X_train, lengths_train)

    train = load_data()
    X_train, y_train, lengths_train = train
    #X_test, y_test, lengths_test = test

    #print("Loading test data...", end=" ")
    #X_test, y_test, lengths_test = load_conll(sys.argv[2], features)
    #describe(X_test, lengths_test)

    clf = StructuredPerceptron(verbose=True, max_iter=10)
    print("Training %s" % clf)
    #print(X_train)
    #print(y_train)
    #print(y_train.shape)
    clf.fit(X_train, y_train, lengths_train)

    joblib.dump(clf, 'model/seq_labeler.pkl') 

    #clf1 = joblib.load('model/seq_labeler.pkl')
    #y_pred = clf1.predict(X_test, lengths_test)
    
    #y_pred = clf.predict(X_test, lengths_test)
    
    #target = codecs.open("../training data/testres", "w", "utf-8")
    
    #for i in range(0, X_test.shape[0]):
        #target.write(y_pred[i]+"\n")
    
    print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
    print("CoNLL F1: %.3f" % (100 * bio_f_score(y_test, y_pred)))