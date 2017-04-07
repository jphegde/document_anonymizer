import os, sys
from glob import glob
import codecs

def main():
    path = "../training data/test data/trainadam_rosa.pdf.txt"#"../training data/ent 2/*.txt"
    filelist = glob( path )
    k = 0
    for i in filelist:
        f = codecs.open(i, "r+b", "utf-8")
        print (i)
        f1 = codecs.open("../training data/labelled test data/resume"+str(k)+".txt", "w", "utf-8")#"../training data/new training/new ent 2/new_"+str(k)+".txt", "w", "utf-8")
        k += 1
        lines = f.readlines()
        for line in lines :
            tokens = line.split(" ")
            #"\n".join(tokens)
            #line.replace(" ", "\n")
            for j in tokens:
                word, sep, tag = j.partition("/")
                f1.write(word+" "+tag+"\n")
                
            
    
if __name__ == "__main__":
    main()