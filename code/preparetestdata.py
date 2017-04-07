import os, sys
from glob import glob
import codecs
import nltk

def main():
    path = "../training data/labelled test data/resume0.txt"
    filelist = glob( path )
    k = 0
    words = []
    for i in filelist:
        f = codecs.open(i, "r+b", "utf-8")
        f1 = codecs.open("../training data/test data/input/test_res_resume_"+str(k)+".txt", "w", "utf-8")
        k += 1
        lines = f.readlines()
        for line in lines :
            tokens = nltk.word_tokenize(line)
            #"\n".join(tokens)
            #line.replace(" ", "\n")
            i = 0
            for j in tokens:
                if i%2 == 0:
                    f1.write(j+" -\n")
                i += 1    
                words.append(j) 
    
    return words
    
if __name__ == "__main__":
    main()