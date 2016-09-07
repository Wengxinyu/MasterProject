# import csv
#
# judgement = dict()
# if __name__ =='__main__':
#     featurefilename = "OriginalData/RTS34F_features.csv"
#     judgementfilename = "OriginalData/RTS34F_judgements.csv"
#     csvfile = open ("Data/RTS34F.csv", "w")
#     with open(judgementfilename) as judgementfile:
#         reader = csv.reader(judgementfile)
#         for row in reader:
#             if judgement.get(row[0]) is None:
#                 judgement[row[0]]=[]
#             judgement[row[0]].append(row[1])
#         # print judgement
#     with open(featurefilename) as featurefile:
#         reader = csv.reader(featurefile)
#         for row in reader:
#             if row[0] in judgement.keys():
#                 if row[1] in judgement[row[0]]:
#                     # print type(row)
#                     writer=csv.writer(csvfile)
#                     writer.writerow(row)
#         csvfile.close()

import csv

judgement = dict()
if __name__ =='__main__':
    featurefilename = "OriginalData/2IRKBE9U_features.csv"
    judgementfilename = "Data/alljudgements.csv"
    csvfile = open ("Data/2IRKBE9U.csv", "w")
    with open(judgementfilename) as judgementfile:
        reader = csv.reader(judgementfile)
        for row in reader:
            if judgement.get(row[0]) is None:
                judgement[row[0]]=[]
            judgement[row[0]].append(row[1])
        # print judgement
    with open(featurefilename) as featurefile:
        reader = csv.reader(featurefile)
        for row in reader:
            if row[0] in judgement.keys():
                if row[1] in judgement[row[0]]:
                    # print type(row)
                    writer=csv.writer(csvfile)
                    writer.writerow(row)
        csvfile.close()


