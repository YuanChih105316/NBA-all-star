import pandas as pd
import glob
import datetime
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import normalize
from sklearn.utils.random import sample_without_replacement
import joblib
#from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import seaborn as sns
import random

np.random.seed(999)
if not os.path.exists('./model'):
    os.makedirs('./model')
if not os.path.exists('./ConfusionMatrix'):
    os.makedirs('./ConfusionMatrix')

plt.rcParams['savefig.dpi'] = 150  # 图片像素
plt.rcParams['figure.dpi'] = 150  # 分辨率

def plot_roc_curve(fpr, tpr, auc, model):
    plt.plot(fpr, tpr, color='orange', label='ROC_' + model)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, AUC: {:3f}'.format(auc))
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.show()

dataset = pd.read_csv('mergedatadup.csv',encoding='utf-8')
dataset.drop(['index','wake','heart_min','rem','floors','heart_avg','HL_TOTAL'],axis =1,inplace = True)

dataset.dropna(inplace=True) 

patienList = list(dataset['id'].unique())

random.seed(126) 
trainPatient = random.sample(patienList, int(len(patienList)*0.8))

valSet =dataset[~dataset['id'].isin(trainPatient)]
trainSet = dataset[dataset['id'].isin(trainPatient)]

valSet.to_csv('datavalidationSet.csv', encoding='utf-8-sig', index=False)
trainSet.to_csv('trainandtestSet.csv', encoding='utf-8-sig', index=False)

trainSet.drop(['id'], inplace=True, axis=1)
valSet.drop(['id'], inplace=True, axis=1)

x_train, x_test = train_test_split(trainSet, test_size=0.3, random_state=1005)

x_train.to_csv('totrain.csv',encoding='utf-8-sig')
x_test.to_csv('totest.csv',encoding = 'utf-8-sig')

x_train.drop('date',inplace=True,axis=1)
x_test.drop('date',inplace=True,axis=1)

print('train: ', x_train.shape[0], '/ obesity true: ', sum(x_train['obesity']))
print('test: ', x_test.shape[0], '/ obesity true: ', sum(x_test['obesity']))
print('val: ', valSet.shape[0], '/ obesity true: ', sum(valSet['obesity']))

y_train=x_train['obesity']
x_train.drop(['obesity'], inplace=True, axis=1)
y_test=x_test['obesity']
x_test.drop(['obesity'], inplace=True, axis=1)

classifier_model = []
classifier_accuracy = []
CM = []
AUC = []                                            #ＲＯＣ曲線：以假陽性率作為x軸，真陽性率為y軸
fpr_tpr = []
pca = PCA()

np.random.seed(999)

from sklearn.ensemble import RandomForestClassifier
classifier_model.append('RandomForest')

#建立預測模型
rfc = RandomForestClassifier()#optional parameters
rfc.fit(x_train, y_train)
classifier_accuracy.append(rfc.score(x_test, y_test))
y_pred = rfc.predict(x_test)

CM.append(confusion_matrix(y_test, y_pred))
joblib.dump(rfc,'./model/RandomForest.pkl')
prob = rfc.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, prob)
AUC.append(auc)
fpr, tpr, thresholds = roc_curve(y_test, prob) #這邊為什麼分三個？
fpr_tpr += [[fpr, tpr]]#+=[[a,b]]
plot_roc_curve(fpr, tpr, auc, 'RandomForest')


plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']  #plt :pyplot

#將資料特徵導入
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)  #estimator?

indices = np.argsort(-importances)[::-1]
#numpy.argsort(a, axis=- 1, kind=None, order=None)
#a:array to sort

# Print the feature ranking
print("Feature ranking:")


for f in range(x_train.shape[1]):
    print("{}.\t {} \t {:4f}" .format(f + 1, x_train.columns[indices[f]], importances[indices[f]]))



#for i,v in enumerate(importances):
#    print('Feature: %0d, Score: %.5f' % (i,v))

# Plot the feature importances of the forest
# plt.figure()
plt.figure()    #這種不用指定變數給他嗎？感覺是生成一個新物件啊
plt.title("Feature importances")
# plt.bar(range(x_train.shape[1]), importances[indices],
#        color="r", align="center")
#-----------------------------------------------------------------------------------------------------------------------------------------------change here#
plt.barh(x_train.columns[indices], importances[indices], align='center')
plt.yticks(x_train.columns[indices], rotation=0, size=6)

result = {}
result = {
    'model':classifier_model,
    'accuracy':classifier_accuracy,
    'confusion matrix':CM,
    'AUROC': AUC
}
result = pd.DataFrame(result)
result['Specificity'] = result['confusion matrix'].apply(lambda x: x[0,0]/(x[0,0]+x[0,1]))
result['Sensitivity'] = result['confusion matrix'].apply(lambda x: x[1,1]/(x[1,1]+x[1,0]))
result['Precision'] = result['confusion matrix'].apply(lambda x: x[1,1]/(x[1,1]+x[0,1]))
result['F1_score'] = 2 / ((1/result['Sensitivity'])+(1/result['Precision']))

result.to_csv('randomforestdatatable.csv', index=False)
