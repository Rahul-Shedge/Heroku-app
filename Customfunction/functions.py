import numpy as np
import pandas as pd
from application_logging import logger
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import roc_auc_score


class Function:

    def __init__(self):
        self.file_object = open("Training_Logs/Training_logs.txt", 'a+')
        self.log_writer  = logger.App_Logger()

    def createdummies(self, listOfCol, data):
        try:
            for col in listOfCol:
                varname         = col.replace('-', '_').replace('?', '').replace(" ", '_') + '_isNan'
                data[varname]   = np.where(pd.isnull(data[col]), 1, 0)
                data.drop([col], 1, inplace=True)
            self.log_writer.log(self.file_object, 'createdummies function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running createdummies function!! Error:: %s' % ex)
            raise ex
        return data 

    def ToNumCat(self, cols, data):
        try:
            data[cols] = np.where(data[cols] == 'Yes', 1, 0)
            self.log_writer.log(self.file_object, 'ToNumCat function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running ToNumCat function!! Error:: %s' % ex)
            raise ex
        return data


    def SortIssue(self, Col, data):
        try:
            k = data[Col].value_counts()
            for val in k.axes[0][0:10]:
                varname = Col + "_" + val.replace(',', '_').replace(' ', '_')
                data[varname] = np.where(data[Col] == val, 1, 0)
            del data[Col]
            self.log_writer.log(self.file_object, 'SortIssue function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running SortIssue function!! Error:: %s' % ex)
            raise ex
        return data


    def clipcols(self, lyst, data):
        try:
            for tr in lyst:
                f = data[tr].value_counts()
                cat = f.index[f > 10000]
                for t in cat:
                    name = tr + '_' + t
                    data[name] = (data[tr] == t).astype(int)
                del data[tr]
            self.log_writer.log(self.file_object, 'clipcols function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running clipcols function!! Error:: %s' % ex)
            raise ex
        return data

    def toDate(self, Colname, data):
        try:
            data[Colname] = pd.to_datetime(data['Date sent to company']) - pd.to_datetime(data['Date received'])
            data[Colname] = data[Colname].astype(str).map(lambda x: x.rstrip('days 00:00:00.000000000').rstrip(' days +'))
            data[Colname] = data[Colname].map(lambda x: x.lstrip())
            data[Colname] = data[Colname].replace(to_replace=[''], value=[0])
            data[Colname] = [int(x) for x in data[Colname]]
            data[Colname] = data[Colname].replace(to_replace=[-1], value=[0])
            data[Colname] = pd.to_numeric(data[Colname], errors="coerce")
            data[Colname] = [int(x) for x in data[Colname]]
            del data['Date received']
            del data['Date sent to company']
            self.log_writer.log(self.file_object, 'toDate function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running toDate function!! Error:: %s' % ex)
            raise ex
        return data

    def Logistic_Model(self,X_train, Y_train, X_test, Y_test):
        try:
            self.file_object = open("Training_Logs/Training_logs.txt", 'a+')
            self.log_writer = logger.App_Logger()
            logr = LogisticRegression(penalty='l2', class_weight="balanced", random_state=3)
            logr.fit(X_train, Y_train)
            print("Trainning ROC-AUC score : " + str(roc_auc_score(Y_train, logr.predict(X_train))))
            print("Testing ROC-AUC score : " + str(roc_auc_score(Y_test, logr.predict(X_test))))
            pickle.dump(logr, open('Models/Logistic_Model.pickle', 'wb'))
            self.log_writer.log(self.file_object, "Logistic_Model Successfully Created & Model is Dumped Successfully ")
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running createdummies function!! Error:: %s' % ex)
            raise ex

