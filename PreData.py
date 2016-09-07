# -*- coding: utf-8 -*-
'''筛选出左右两列所有的documents'''

# import sys
# import csv
#
#
# judgements = dict()
# features = dict()
# if __name__=='__main__':
#     featurefilename = 'Data/allfeatures.csv'
#     judgefilename = 'Data/alljudgements.csv'
#     outputfile = open('Data/features_judgements.csv','w')
#     with open(judgefilename) as judgefile:
#         reader = csv.reader(judgefile)
#         for line in reader:
#             if judgements.get(line[0]) is None:
#                 judgements[line[0]]={}
#             judgements[line[0]][line[1]]=line[2]
#         # for line in reader:
#             if judgements.get(line[1])is None:
#                 judgements[line[1]]={}
#             judgements[line[1]][line[0]]=line[2]
#     # print len(judgements)
#     # print judgements
#     with open(featurefilename) as featurefile:
#         reader = csv.reader(featurefile)
#         # for row in reader:
#         #     # print row
#         #     documents = judgements[row[0]]
#         #     label = documents[row[1]]
#         #     # print ','.join(row)
#         #     outputfile.write(str(label)+','+','.join(row)+'\n')
#         # outputfile.close()
#         for row in reader:
#             if features.get(row[0]) is None:
#                 features[row[0]]={}
#             features[row[0]][row[1]]=row[2:]
#         # for row in reader:
#             if features.get(row[1]) is None:
#                 features[row[1]]={}
#             features[row[1]][row[0]]=row[2:]
#
#         # print len(features)
#     for document1 in features:
#         docs = judgements[document1] #the second column documents
#         for document2 in features[document1]:
#             # print features[document1][document2]
#             label = docs[document2]
#             s = ','.join(features[document1][document2])
#             outputfile.write(str(label)+','+str(document1)+','+str(document2)+','+s+'\n')
#     outputfile.close()

'''找出第一列文档有可能在第二列的情况(querydocuments)'''
# import sys
# import csv
#
#
# judgements = dict()
# features = dict()
# if __name__=='__main__':
#     featurefilename = 'Data/alldocuments.csv'
#     judgefilename = 'Data/alljudgements.csv'
#     outputfile = open('Data/querydocuments.csv','w')
#     with open(judgefilename) as judgefile:
#         reader = csv.reader(judgefile)
#         for line in reader:
#             if judgements.get(line[0]) is None:
#                 judgements[line[0]]={}
#             judgements[line[0]][line[1]]=line[2]
#         # # for line in reader:
#         #     if judgements.get(line[1])is None:
#         #         judgements[line[1]]={}
#         #     judgements[line[1]][line[0]]=line[2]
#     # print len(judgements)
#     # print judgements
#     with open(featurefilename) as featurefile:
#         reader = csv.reader(featurefile)
#         for row in reader:
#             if row[1] in judgements.keys():
#                 writer = csv.writer(outputfile)
#                 writer.writerow(row)
#         outputfile.close()

'''csv转为txt'''

# import csv
#
# if __name__=='__main__':
#     csvfilename = 'Data/querydocuments_label3.csv'
#     txtfile = open('Data/querydocuments_label3.txt','w')
#     with open(csvfilename) as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             txtfile.write(','.join(line)+'\n')
#     txtfile.close()

# '''文件名映射为数字'''
#
# if __name__=='__main__':
#     txtfilename = 'Data/querydocuments_label2.txt'
#     newfile = open('Data/qdnumber2.txt','w')
#     with open(txtfilename) as txtfile:
#
#         for line in txtfile:
#             line = line.strip().split(',')
#             line[1]=str(hash(line[1]))
#             line[2]=str(hash(line[2]))
#             print line
#             newfile.write(','.join(line)+'\n')
#
#     newfile.close()

'''写成适合letor格式的'''

if __name__=='__main__':
    txtfilename = 'Data/querydocuments_label3.txt'
    newfile = open('Data/qdnumber3.txt','w')
    with open(txtfilename) as txtfile:
        for line in txtfile:
            line = line.strip().split(',')
            l = len(line)
            # print line[l-1]
            line[2],line[l-1]=line[l-1],line[2]
            s = []
            for i in range(l-1):
                if i == 0:
                    s.append(line[i])
                elif i == 1:
                    s.append('qid' + ':' + str(line[i]))
                elif i == l-1:
                    s.append('#' + str(line[i]))
                else:
                    s.append(str(i-1) + ':' + str(line[i]))
            newfile.write(' '.join(s)+'\n')
    newfile.close()

