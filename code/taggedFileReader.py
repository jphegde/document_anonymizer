import codecs

def readFile(filename):
    #f = codecs.open(filename, "r", "utf-8")
    f = open(filename, "r")
    lines = f.readlines()
    
    words = []
    tags = []
    
    for line in lines:
        tokens = line.split(" ")
        #print tokens
        for token in tokens:
            elems = token.partition('/')
            #(word, tag) = token.split("/", 1)
            words.append(elems[0])
            tags.append(elems[2])
            
    return words, tags
           