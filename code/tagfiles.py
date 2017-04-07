import tagging
import taggedFileReader
import os, sys

def main():
    path = "../data/design resumes/text_resumes/"
    filelist = os.listdir( path )
    j = 0
    for i in filelist:
        tagging.create_tagFile(path+filelist[j], "../tagged training data/tagged resumes/train"+filelist[j])
        #taggedFileReader.readFile("../tagged training data/bbc/business/train"+filelist[i]+".txt")
        j += 1
    
if __name__ == "__main__":
    main()