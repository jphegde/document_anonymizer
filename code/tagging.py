import nltk
import codecs

def create_tagFile(filename, dest):
    f = codecs.open(filename, "r", "utf-8")
    lines = f.readlines()
    target = codecs.open(dest, "w", "utf-8")
    
    for line in lines:
        tokens = nltk.word_tokenize(line)
        tagged_tokens = nltk.pos_tag(tokens)
        for sent in tagged_tokens:
            target.write(nltk.tag.tuple2str(sent)+" ")
        #target.write(str(tagged_tokens))