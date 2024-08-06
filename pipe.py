import pandas as pd
import numpy as np
import dill
import logging
import missingno

from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
from my_module__stat import *

from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import cross_validate,cross_val_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler 
import optuna
import pickle

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from logging import warning
from datetime import datetime


def data_transform(data):
    cat_cols = data.select_dtypes('object').drop(columns='session_id').columns
    data.loc[~data['utm_source'].isin(data['utm_source'].value_counts().head(10).index),'utm_source'] = 'other'
    data.loc[~data['utm_campaign'].isin(data['utm_campaign'].value_counts().head(7).index),'utm_campaign'] = 'other'
    data.loc[~data['utm_keyword'].isin(data['utm_keyword'].value_counts().head(3).index),'utm_keyword'] = 'other'
    data.loc[~data['device_brand'].isin(data['device_brand'].value_counts().head(6).index),'device_brand'] = 'other'
    data.loc[~data['geo_city'].isin(data['geo_city'].value_counts().head(5).index),'geo_city'] = 'other'

    del data['geo_country']

    cat_cols = list(cat_cols)

    cat_cols.pop(-2)

    numerical = data.select_dtypes('number').drop(columns='target')
    categorical = data.select_dtypes('object').drop(columns='session_id')

    scaler = MinMaxScaler()

    for col in numerical.columns:

        data[col] = scaler.fit_transform(data[[col]])


    to_encode = categorical.columns

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(data[to_encode])

    data[ohe.get_feature_names_out()] = ohe.transform(data[to_encode])

    data = data.drop(columns=categorical.columns).drop(columns='session_id')

    return data


def fillna_organize_data(df):

    def mul(s):
        l = s.split('x')
        return int(l[0])*int(l[1])

    df['utm_campaign'] = df['utm_campaign'].fillna('no_campaign')
    df['utm_adcontent'] = df['utm_adcontent'].fillna('no_adcontent')
    df['utm_keyword'] = df['utm_keyword'].fillna('no_keyword')
    df['device_os'] = df['device_os'].fillna('unknown_device')
    df['device_brand'] = df['device_brand'].fillna('unknown_device_brand')
    del df['device_model']

    df = df.drop(columns=['client_id','visit_date'])

    df['visit_time'] = pd.to_datetime(df.visit_time).dt.hour
    df.loc[~df['utm_source'].isin(df['utm_source'].value_counts().head(20).index),'utm_source'] = 'other'
    df.loc[~df['utm_medium'].isin(df['utm_medium'].value_counts().head(10).index),'utm_medium'] = 'other'
    df.loc[~df['utm_campaign'].isin(df['utm_campaign'].value_counts().head(30).index),'utm_campaign'] = 'other'
    df.loc[~df['utm_adcontent'].isin(df['utm_adcontent'].value_counts().head(15).index),'utm_adcontent'] = 'other'
    df.loc[~df['utm_keyword'].isin(df['utm_keyword'].value_counts().head(30).index),'utm_keyword'] = 'other'
    df.loc[~df['device_os'].isin(df['device_os'].value_counts().head(6).index),'device_os'] = 'other'
    df.loc[~df['device_brand'].isin(df['device_brand'].value_counts().head(10).index),'device_brand'] = 'other'
    df.loc[~df['device_browser'].isin(df['device_browser'].value_counts().head(7).index),'device_browser'] = 'other'
    df.loc[~df['geo_country'].isin(df['geo_country'].value_counts().head(3).index),'geo_country'] = 'other'
    df.loc[~df['geo_city'].isin(df['geo_city'].value_counts().head(50).index),'geo_city'] = 'other'

    df['screen_square'] = df['device_screen_resolution'].apply(lambda x:mul(x))
    del df['device_screen_resolution']
    df.rename(columns={'event_action_bool':'target'},inplace=True)

    return df

    
def main():

    ga_hits = pd.read_csv('34_Finals/ga_hits.csv'#,nrows=1000000
    )
    ga_sessions = pd.read_csv('34_Finals/ga_sessions.csv'#,nrows=1000000
    )

    ga_hits['target'] = ga_hits['event_action'].isin(['sub_car_claim_click', 'sub_car_claim_submit_click',
    'sub_open_dialog_click', 'sub_custom_question_submit_click',
    'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
    'sub_car_request_submit_click'])*1

    grouped = ga_hits[['session_id','hit_number','target']].groupby('session_id').agg(max).reset_index()
    grouped = grouped[['session_id','target']]
    merged = ga_sessions.merge(grouped, on='session_id',how='inner')[list(ga_sessions.columns) + ['target']]

    df = merged.copy()

    df = fillna_organize_data(df)

    cat_cols = df.select_dtypes('object').drop(columns='session_id').columns


    X = df.drop(['target'], axis=1)
    y = df['target']

    random_state=42

    OverS = RandomOverSampler(random_state=42)

    X_over, y_over = OverS.fit_resample(X, y)
    df_over = X_over.copy()
    df_over['target'] = y_over

    df_over = data_transform(df_over)

    df_ml = df_over.copy()
    X_train = df_ml.drop(['target'], axis=1)
    y_train = df_ml['target']

    lc = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    lc.fit(X_train, y_train)

    xc = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xc.fit(X_train, y_train)

    clf_list = [
        # (log, "Logistic Regression"),
        # (gnb, "Naive Bayes"),
        # (cbc, "CatBoostClassifier"),
        (lc,'LGBMClassifier'),
        # (rf, "Random forest"),
        (xc, "XGBClassifier"),
    ]
    clf_list_r = [(x[1],x[0]) for x in clf_list]

    eclf2 = VotingClassifier(estimators=clf_list_r, voting='soft')

    eclf2 = eclf2.fit(X_train, y_train)


    y_proba = xc.predict_proba(X_train)[::,1]
    fpr, tpr, _ = roc_curve(y_train,  y_proba)
    # auc = roc_auc_score(y_train, y_proba)

    # warning('auc_value =', auc)

    with open('31_model_as_api/price_cat.pkl', 'wb') as file:
        dill.dump({
            'model': eclf2,
            'metadata': {
                'name': 'sber_hit_predict',
                'author': 'Alexander Nisilevich',
                'version': 1,
                'date': datetime.now(),
                'type': 'LGBMClassifier + XGBClassifier',
                'auc': 0.694
                }
                }
                ,file, recurse=True)


if __name__ == '__main__':
    main()
