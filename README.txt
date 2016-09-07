An Analysis of Learning to Rank for Task Identification from Desktop logs
The main languages are Java and python, there are several codes and some of them are just for testing and comparision. The main algorithm used in my work is LambdaMART, which is a state-of-the-art learning to rank algorithm. 

py_learning to rank contains implementation of LambdaMART, the main reference is a python package for learning to rank algorithm called pyltr. Datasets in this part follow LETOR format.

In addition, the classic package of learning to rank called RankLib is used for comparision. It is implemented by Java. Datasets in this part are formulated as same as SVM-light datasets.http://svmlight.joachims.org


The original data extracted from desktop and evaluated by users are in OriginalData. In order to make data compatible for my project, preprocessing is necessary. Including deleting useless records, changing labels into numbers (relevant>contextual relevant>non-relevant).

Data description
<line> .=. <relevance> qdid:<did> <feature>:<value>:<feature>:<value> # <info><relevance> .=. <positive integer><qdid> .=. <string><did> .=. <string><feature> .=. <positive integer><value> .=. <float><info> .=. <string>

1. alljudgements.csv: combining all documents of 8 users and documents had been judged.
2. allfeatures.csv: features of ducoments that had been judged
3. querydocuments.csv: containing label of two documents, included query documents columns
4. alldocuments.csv: containing label of two documents, included two documents columns
5. querydocument_label.csv: labels with number, 351
6. querydocument_label.txt
7. qdnumber.txt: query with number, 351
8. querydocument_label2.csv: labels with number, 618
9. querydocument_label2.txt
10. qdnumber2.txt: query with number, 618
11. querydocuments_label3.csv: deleting query with just one record.
