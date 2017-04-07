
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



import sequence_learner_new

def test_model():
    
    print("Loading test data...", end=" ")
    test_files = [f for i, f in enumerate(glob("prepard_test_1.txt"))]
    test = load_conll(fileinput.input(test_files), sequence_learner_new.features)
    X_test, _, lengths_test = test
    sequence_learner_new.describe(X_test, lengths_test)
    
    X_test, y_test, lengths_test = test
    Y =  sequence_learner_new.get_labels(y_test)
    clf = joblib.load('model/seq_labeler.pkl')
    y_pred = clf.predict(X_test, lengths_test)
    
    target = codecs.open("test_res.txt", "w", "utf-8")
    
    for i in range(0, X_test.shape[0]):
        target.write(y_pred[i]+"\n")
        
    prec = 0
    recall = 0
    count = 0
    entities = ["<PER>", "</PER>", "<PER></PER>", "<IPER>", "<ORG>", "</ORG>", "<ORG></ORG>", "<IORG>"]
    for i in range(len(y_test)):
        elems = Y[i].partition("<")
        entity = elems[1]+elems[2]
        entity = entity.replace('\n', '')
        yelem = y_pred[i].partition("<")
        yent = yelem[1]+yelem[2]
        
        if entity in entities and Y[i].replace('\n', '') == y_pred[i]:
            print (entity)
            count += 1
            prec +=1
            
        elif yent in entities and Y[i] != y_pred[i]:
            count += 1
    
    
    if count > 0:
        print (count)
        print( " precision = ",(float(prec)/float(count))*100,"%")
    else:
        print( " precision = 0" , prec)
        
    print("Accuracy: %.3f" % (100 * accuracy_score(Y, y_pred)))
        
        
def train_model():
    train = sequence_learner_new.load_data()
    X_train, y_train, lengths_train = train
    Y = sequence_learner_new.get_labels(y_train)
    clf = joblib.load('new model/new_seq_labeler.pkl')
    clf.fit(X_train, Y, lengths_train)

    joblib.dump(clf, 'new model/new_seq_labeler.pkl') 
    
    
def main():
    #train_model()
    test_model()
    
if __name__=="__main__":
    main()
