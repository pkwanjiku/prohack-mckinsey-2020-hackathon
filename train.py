# import libs
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from scipy.optimize import minimize

import warnings 
warnings.filterwarnings('ignore')

import datawig


# import training data
pd.set_option('display.max_columns',None)

# training data
train = pd.read_csv('train.csv')
train = train.dropna()


# test data
test = pd.read_csv('test.csv')
df=pd.concat([train,test], sort=False)
train.head()

df["galaxy"] = df["galaxy"].astype('category')
df["galaxy"] = df["galaxy"].cat.codes
train = df[:3865]
test = df[3865:]
test=test.drop("y", axis = 1)
test_res= test.copy()


# Checking how many galaxies are there and how many of them are distinct.
# There are 181 distinct galaxies on the training set and 172 on the test set.

# On overall they each galaxy has 20 samples on the training set and 5 on the test set.

# Some galaxies on the training set does not exist on the test set.

# Galaxy 126 has only one sample. I discard it on the training phase


train_gal=set(train["galaxy"])
s=0
for x in train_gal:
    s=s+len(train.loc[train['galaxy'] == x])
print("Total distinct galaxies: {}".format(len(train_gal)))
print("Average samples per galaxy: {}".format(s/len(train_gal)))

# Total distinct galaxies: 181
# Average samples per galaxy: 21.353591160220994

test_gal=set(test["galaxy"])
s=0
for x in test_gal:
    s=s+len(test.loc[test['galaxy'] == x])
print("Total distinct galaxies: {}".format(len(test_gal)))
print("Average samples per galaxy: {}".format(s/len(test_gal)))


print("Train vector: " + str(train.shape))
print("Test vector: " + str(test.shape))


def cross_validation_loop(data,cor):
    labels= data['y']
    data=data.drop('galaxy', axis=1)    
    data=data.drop('y', axis=1)
    
    correlation=abs(data.corrwith(labels))
    columns=correlation.nlargest(cor).index
    data=data[columns]
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data)
    data=imp.transform(data)

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
        
    estimator = GradientBoostingRegressor(n_estimators=300)
    
    cv_results = cross_validate(estimator, data, labels, cv=4, scoring='neg_root_mean_squared_error')

    error=np.mean(cv_results['test_score'])
    
    return error

train_gal=set(train["galaxy"])
train_gal.remove(126)
def loop_train(cor):
    errors=[]
    for gal in train_gal:
        index = train.index[train['galaxy'] == gal]
        data = train.loc[index]
        errors.append(cross_validation_loop(data,cor))
    return np.mean(errors)



# cor=[20,25,30,40,50,60,70,80]
# errors=[]
# for x in cor:
#     errors.append(loop_train(x))

# print(errors)


def test_loop(data, test_data):
    labels= data['y']
    data=data.drop('galaxy', axis=1)    
    data=data.drop('y', axis=1)
    correlation=abs(data.corrwith(labels))
    columns=correlation.nlargest(10).index
    
    train_labels= labels
    train_data=data[columns]
    test_data= test_data[columns]
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(train_data)
    train_data=imp.transform(train_data)
    test_data=imp.transform(test_data)

    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    model = GradientBoostingRegressor(n_estimators=300)
    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    return predictions

test=test_res
test=test.sort_values(by=['galaxy'])
test_pred = pd.DataFrame(0, index=np.arange(len(test)), columns=["predicted_y"])

i=0
for gal in test_gal:
    count=len(test.loc[test['galaxy'] == gal])
    index = train.index[train['galaxy'] == gal]
    data = train.loc[index]
    pred=test_loop(data,test.loc[test['galaxy']==gal])
    test_pred.loc[i:i+count-1,'predicted_y'] = pred
    i=i+count


test["predicted_y"]=test_pred.to_numpy()
test.sort_index(inplace=True)
predictions = test["predicted_y"]

index = predictions
pot_inc = -np.log(index+0.01)+3

p2= pot_inc**2


ss = pd.DataFrame({
    'Index':test.index,
    'pred': predictions,
    'opt_pred':0,
    'eei':test['existence expectancy index'], # So we can split into low and high EEI galaxies
})

ss.loc[p2.nlargest(400).index, 'opt_pred']=100
ss=ss.sort_values('pred')
ss.iloc[400:600].opt_pred = 50
ss=ss.sort_index()

increase = (ss['opt_pred']*p2)/1000

print(sum(increase), ss.loc[ss.eei < 0.7, 'opt_pred'].sum(), ss['opt_pred'].sum())

ss[['Index', 'pred', 'opt_pred']].to_csv('submission.csv', index=False)