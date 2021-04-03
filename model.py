import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from Customfunction import functions

cc_train = pd.read_csv('data/Consumer_Complaints_train.csv')
cc_test = pd.read_csv('data/Consumer_Complaints_test_share.csv')

cc_test["data"] = 'test'
cc_test["Consumer disputed?"] = None
cc_train['data'] = 'train'
cc = pd.concat([cc_train, cc_test], 0, ignore_index=True)

ColList = ['Sub-product', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Tags',
           'Consumer consent provided?']
binaryCatCol = ['Consumer disputed?', 'Timely response?']
feat = ['Company', 'Product', 'Submitted via', 'State']

cd = functions.Function()

cc = cd.createdummies(ColList, cc)
cc = cd.ToNumCat(binaryCatCol, cc)
lp_dummies = pd.get_dummies(cc['Company response to consumer'])
cc = pd.concat([lp_dummies, cc], 1)
#print(cc.columns)
cc = cc.drop(['Company response to consumer'], 1)
cc = cd.SortIssue('Issue', cc)
cc = cc.drop(['ZIP code', 'Complaint ID'], 1)
cc = cd.clipcols(feat, cc)
cc = cd.toDate("Days", cc)
#print(cc.columns)

data_train = cc[cc["data"] == "train"]
data_test = cc[cc["data"] == "test"]
data_test = data_test.drop(["Consumer disputed?"], 1)

data1 = data_train.drop(["data"], 1)
data2 = data_test.drop(["data"], 1)

cc_train1, cc_train2 = train_test_split(data1, test_size=0.2, random_state=2)

x_train = cc_train1.drop(['Consumer disputed?'], 1)
y_train = cc_train1["Consumer disputed?"]
x_test = cc_train2.drop(['Consumer disputed?'], 1)
y_test = cc_train2['Consumer disputed?']

cd.Logistic_Model(x_train, y_train, x_test, y_test)
