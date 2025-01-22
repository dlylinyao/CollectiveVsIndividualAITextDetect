StanfordCoreNLP: 
- cd D:\ProgramFiles\stanford-corenlp-4.5.7
- java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
- java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &

Credits to
``https://github.com/onikula/thesis_files``
