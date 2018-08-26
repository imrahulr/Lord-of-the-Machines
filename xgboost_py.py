import pandas as pd
import numpy as np
import time, re, string
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm_notebook as tqdm

train = pd.read_csv('input/train_featv3.csv')
test = pd.read_csv('input/test_featv3.csv')
print (train.head())

train['time'] = train['hour']*60 + train['mins']
test['time'] = test['hour']*60 + test['mins']

Y_train = train['is_click'].values

cols = ['user_id', 'date', 'time', 'communication_type', 'total_links',  
        'no_of_internal_links', 'no_of_images', 'count_sent', 'count_letters', 'count_punctuations', 
        'count_stopwords', 'word_unique_percent', 'punct_percent', 'email_count_word', 
        'email_count_unique_word', 'email_count_punctuations', 'email_cap_count', 'day_of_week', 
        'count_click', 'count_user', 'click_confidence','count_is_open','is_open_confidence', 
        'body_polarity', 'title_polarity','body_subjectivity', 'title_subjectivity', 
        'email_3_similar', 'sub_3_similar', 'sub_period', 'comm_type_click_percent']

X_train = train[cols]
X_test = test[cols]


## filled new user with click confidence of mean of first users ###
# X_test = X_test.fillna(0.0072169867589168555)
X_test['click_confidence'] = X_test['click_confidence'].fillna(0.0072169867589168555)
X_test['is_open_confidence'] = X_test['is_open_confidence'].fillna(0.10831444590242156)
X_test.loc[X_test['count_user'] == 0,'count_click'] = 0.0072169867589168555
X_test.loc[X_test['count_user'] == 0,'count_is_open'] = 0.10831444590242156
X_test.loc[X_test['count_user'] == 0,'count_user'] = 1

X_train.loc[:,'count_user_freq'] = (1/X_train.loc[:,'count_user'])
X_test.loc[:,'count_user_freq'] = (1/X_test.loc[:,'count_user'])

dropcols = ['body_polarity', 'title_polarity','body_subjectivity', 'title_subjectivity']
X_train.loc[:,'sentiment'] = X_train.loc[:,'body_polarity'] + X_train.loc[:,'title_polarity'] + X_train.loc[:,'body_subjectivity'] + X_train.loc[:,'title_subjectivity']
X_test.loc[:,'sentiment'] = X_test.loc[:,'body_polarity'] + X_test.loc[:,'title_polarity'] + X_test.loc[:,'body_subjectivity'] + X_test.loc[:,'title_subjectivity']
X_train.drop(dropcols,axis=1,inplace=True)
X_test.drop(dropcols,axis=1,inplace=True)

print (X_train.head)


from imblearn.under_sampling import (AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours)

print('RENN')
enn = RepeatedEditedNearestNeighbours(return_indices=True)
X_res, Y_res, idx_res = enn.fit_sample(X_train, Y_train)
reduction_str = ('Reduced {:.2f}%'.format(100 * (1 - float(len(X_res))/len(X_train))))
print (reduction_str)

print (X_res.shape, Y_res.shape)
print (Y_res.sum(), Y_train.sum())


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_rs_trn, X_rs_val, Y_res_trn, Y_res_val = train_test_split(X_res, Y_res, test_size=0.075, shuffle=True
                                                            , random_state=42)

print ("Train_shape: " + str(X_rs_trn.shape))
print ("Val_shape: " + str(X_rs_val.shape))
print ("No of positives in train: " + str(Y_res_trn.sum()))
print ("No of positives in val: " + str(Y_res_val.sum()))


params = {}
params['booster'] = 'gbtree'
params['objective'] = 'binary:logistic'
params['silent'] = 1
params['eta'] = 0.01
params['eval_metric'] = 'auc'
params['max_depth'] = 3
params['colsample_bytree'] = 0.8
params['subsample'] = 0.8
# params['min_child_weight'] = 5

d_train = xgb.DMatrix(X_rs_trn, label=Y_res_trn)
d_valid = xgb.DMatrix(X_rs_val, label=Y_res_val)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

# Third value is number of rounds (n_estimators), early_stopping_rounds stops training when it hasn't 
# improved for that number of rounds
clf = xgb.train(params, d_train, 7200, watchlist, early_stopping_rounds=50, verbose_eval=25)


from operator import itemgetter

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance

print(get_importance(clf, list(X_train.columns.values)))


d_test = xgb.DMatrix(X_test)
d_test.feature_names = d_train.feature_names
p_test = clf.predict(d_test) 

sub = pd.read_csv('input/sample_submission.csv')
sub['is_click'] = p_test
sub.head()
sub.to_csv('sub_xgb_py.csv', index=False)

print (sub.describe())

