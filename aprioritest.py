# Testing the apriori association rule learning algorithm to find commonly purchased-together items from a 
# retail dataset. 

# this uses the apyori module found at https://github.com/ymoch/apyori/blob/master/apyori.py
# and the dataset from https://www.kaggle.com/puneetbhaya/online-retail


# Importing the libraries
import numpy as np
import pandas as pd

######## Data Preprocessing
dataset = pd.read_excel('Online Retail.xlsx')

#Replace null descriptions with the stock code 
for i, d in dataset[dataset['Description'].isnull()].iterrows():
    dataset['Description'][i] = "Code-" + str(d['StockCode'])
#group into baskets
grouped = dataset.groupby('InvoiceNo')

#rearrange into a list, as required by apyori
transactions = []
for name,group in grouped:
    transactions.append(list(group['Description'].map(str)))

#here I am sampling the contents of large baskets, the algorithm runs really long if not. 
np.random.seed(42)
tsamples = []
for row in transactions:
    if len(row) >= 150:
        newrow = np.random.choice(row,size=150).tolist()
    else:
        newrow = row
    tsamples.append(newrow)


##### Training Apriori on the dataset
from apyori import apriori
rules = apriori(tsamples, min_support = 0.005, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Getting the results
results = list(rules)


#results print out courtesy of:
#https://www.kaggle.com/asamir/online-retail-analyze-with-association-rules
final_results = pd.DataFrame(np.random.randint(low=0, high=1, size=(len(results), 6)), columns=['GeneralRules', 'LeftRules', 'RightRules', 'Support', 'Confidence', 'Lift'])
index = 0
for g, s, i in results:
    final_results.iloc[index] = [' _&&_ '.join(list(g)), ' _&&_ '.join(list(i[0][0])), ' _&&_ '.join(list(i[0][1])), s, i[0][2], i[0][3]]
    index = index+1
# The most significant rules
final_results = final_results.sort_values('Lift', ascending=0)
#final_results.head(20)

count=1
for i, d in final_results.head(20).iterrows():
    print('Rule #'+str(count)+':')
    print(d['LeftRules'])
    print('=> '+d['RightRules'])
    print('Support: '+str(d['Support'])+' - Confidence: '+str(d['Confidence'])+' - Lift: '+str(d['Lift']))
    print('--------------------')
    count=count+1