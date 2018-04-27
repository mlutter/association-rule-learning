




## This uses pyfim from http://www.borgelt.net/pyfim.html

# and the dataset from https://www.kaggle.com/puneetbhaya/online-retail


# Importing the libraries
import numpy as np
import pandas as pd
from fim import eclat
help(eclat)


######## Data Preprocessing
dataset = pd.read_excel('Online Retail.xlsx')

#Replace null descriptions with the stock code 
for i, d in dataset[dataset['Description'].isnull()].iterrows():
    dataset['Description'][i] = "Code-" + str(d['StockCode'])
#group into baskets
grouped = dataset.groupby('InvoiceNo')

#rearrange into a list
transactions = []
for name,group in grouped:
    transactions.append(list(group['Description'].map(str)))

##### Training 
report = eclat(transactions, target='s', supp=1, zmin=2)

##compare with apriori from same module. Different from apyori.... maybe have to investigate further
from fim import apriori
help(apriori)
areport = apriori(transactions, report='l,c', target='r', supp=1, zmin=2)