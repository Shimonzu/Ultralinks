import pandas as pd
import numpy as np
from collections import Counter
import operator
from datetime import datetime
from datetime import date, timedelta
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
import seaborn as sns
from statsmodels.discrete.discrete_model import Logit
import statsmodels.tools as smt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from imblearn.over_sampling import SMOTE
from confumatrix import ConfuMatrix


#This will load the data from your working directory, and create two data tables: 1. Hourly user checkins (done automatically).
#2. Click data of users. Originally, the click data was going to be used to extend insight from the checkin table,
#however, without a good way to link the data from the two tables by user, this had to be abandoned.
def load_data():
    checkins = pd.read_csv("checkins.csv")
    clicks = pd.read_csv("clicks.csv")
    return checkins, clicks

#This function converts timestamp data into formats that are easier to work with.
def convert_timestamps(checkins, clicks):
    #Converts times to timestamps.
    #Removes data points that were created during product testing.
    checkins['time']= pd.to_datetime(checkins['time'])
    checkins['date'] = checkins['time'].apply(lambda x:x.date().strftime('%Y %m %d'))
    clicks['Time']= pd.to_datetime(clicks['Time'])
    clicks['date'] = clicks['Time'].apply(lambda x:x.date().strftime('%Y %m %d'))
    checkins = checkins[checkins['account']!=2]
    checkins = checkins[checkins['account']!=30]
    return checkins, clicks

#Determines the earliers checkin by user, and considers that the user registration date. The last registration date is used
#subsequently to determine if a user has churned or not.
def create_regs_last_checkins(checkins, clicks):
    registrations = pd.DataFrame(checkins.groupby(['uuid'], as_index=False)['time'].min())
    registrations.columns = ['uuid', 'registration']
    registrations['reg_date'] = registrations['registration'].apply(lambda x:x.date().strftime('%Y %m %d'))
    registrations['num'] = 1

    last_checkins = pd.DataFrame(checkins.groupby(['uuid'], as_index=False)['time'].max())
    last_checkins.columns = ['uuid', 'last_checkin']
    last_checkins['last_checkin_date'] = last_checkins['last_checkin'].apply(lambda x:x.date().strftime('%Y %m %d'))

    return registrations, last_checkins

#Creates the master_data table which will be used to do analysis. This funtion combines the registrations, last_checkins. Originally,
#it also combined click data, but since the IP address was determined to be an invalid method of joining these, it is not used
#here.
def create_master_data(registrations, last_checkins, checkins, clicks):
    master_data = pd.merge(registrations, last_checkins, on = 'uuid')
    master_data['time_active'] = master_data['last_checkin'] - master_data['registration']
    master_data['time_active_in_sec'] = master_data['time_active'] / np.timedelta64(1,'s')
    master_data['time_active_in_days'] = master_data['time_active'] / np.timedelta64(1,'D')


    master_data['last_checkin_date_only'] = master_data['last_checkin'].apply(lambda x:x.date().strftime('%Y %m %d'))
    master_data['time_active<1hour'] = master_data['time_active_in_sec'] <= 3600
    master_data['time_active<1day'] = master_data['time_active_in_sec'] <= 86400

    dayofdatapull = max(checkins['time'])
    thirtydaysago = dayofdatapull - timedelta(days=30)

    master_data['age'] = dayofdatapull - master_data['registration']
    master_data['age_in_days'] = master_data['age']/ np.timedelta64(1,'D')
    master_data['age_in_hours'] = master_data['age']/ np.timedelta64(1,'h')
    master_data['days_since_last_checkin'] = dayofdatapull - master_data['last_checkin']
    master_data['churn'] = master_data['last_checkin'] < thirtydaysago #Check to see if this is the correct direction!!!!
    master_data['churn'] = master_data['churn'].astype(int)
    master_data['days_since_last_checkin'] = dayofdatapull - master_data['last_checkin']
    master_data['inactive1'] = master_data['days_since_last_checkin'] < timedelta(days=1)
    master_data['inactive1-7'] = (master_data['days_since_last_checkin'] >= timedelta(days=1)) & (master_data['days_since_last_checkin'] < timedelta(days=7))
    master_data['inactive7-15'] = (master_data['days_since_last_checkin'] >= timedelta(days=7)) & (master_data['days_since_last_checkin'] < timedelta(days=15))
    master_data['inactive15-30'] = (master_data['days_since_last_checkin'] >= timedelta(days=15)) & (master_data['days_since_last_checkin'] < timedelta(days=30))
    master_data['inactive1'] = master_data['inactive1'].astype(int)
    master_data['inactive1-7'] = master_data['inactive1-7'].astype(int)
    master_data['inactive7-15'] = master_data['inactive7-15'].astype(int)
    master_data['inactive15-30'] = master_data['inactive15-30'].astype(int)

    churns = master_data[master_data['churn'] == True]

    checkin_uuids = set(checkins['uuid'].unique())
    click_uuids = set(clicks['UUID'].unique())

    #This removes right censored data. Because it takes 30 days to fully churn, anybody that isn't at least 30 days old must be removed
    #because they would automatically be classified as non-churn when we don't know that yet.
    return master_data, churns, click_uuids, checkin_uuids

#The checkin table had a number of feature variables that needed to be converted into different data formats in order for
#my models to make meaningful predictions. This function convdrts these features into those appropriate formats.
def add_additional_features_to_master_data(checkins, master_data, last_checkins):
    authorizations = checkins[['twitterAuth','linkedinAuth','facebookAuth', 'googleplusAuth', 'angellistAuth', 'uuid']]
    authorizations = authorizations.groupby('uuid').max()[['twitterAuth','linkedinAuth','facebookAuth', 'googleplusAuth', 'angellistAuth']]
    authorizations['uuid'] = authorizations.index
    #I'm using "max" here because I want to know if the authorizations have ever been done.

    master_data = pd.merge(master_data, authorizations, how = 'left', left_on='uuid', right_on='uuid')


    additional_features = checkins[['uuid','extensionVersion', 'browser', 'browserVersion', 'OS', 'ultralinksEnabled', 'combineSimilarButtons', 'replaceHyperlinks', 'multipleSearchOptions', 'newWindowLinks', 'proximityFade', 'newInlineUltralinks', 'hoverTime', 'popupRecoveryTime', 'iconAlignment', 'fragmentCFMiss', 'fragmentCFHit', 'language1', 'language2', 'language3', 'country1', 'country2', 'country3', 'whiteList', 'prerelease', 'extensionID']]
    #Removed pages scanned and pages browsed. This is probably the same thing as time_active, and it also doesn't say much about the user experience.
    additional_features[['Chrome', 'Firefox', 'Opera', 'Opera Next', 'Safari']] = pd.get_dummies(additional_features['browser'])
    additional_features[['Android','Linux', 'Mac', 'WINNT', 'Windows']] = pd.get_dummies(additional_features['OS'])


    browser_OS = additional_features[['Chrome', 'Firefox', 'Opera', 'Opera Next', 'Safari', 'Android','Linux', 'Mac', 'WINNT', 'Windows', 'uuid']].groupby('uuid').max()
    browser_OS['uuid'] = browser_OS.index
    #For browser OS, I'm finding the browswer that the user used most.
    ulink_features = additional_features[['uuid', 'ultralinksEnabled', 'combineSimilarButtons', 'replaceHyperlinks', 'multipleSearchOptions', 'newWindowLinks', 'proximityFade', 'newInlineUltralinks']]
    ulink_features = ulink_features.groupby('uuid').max()
    ulink_features['uuid'] = ulink_features.index
    average_hover_time = additional_features.groupby('uuid').mean()['hoverTime']
    average_hover_time = pd.DataFrame(average_hover_time)
    average_hover_time['uuid'] = average_hover_time.index
    #Because hover time has to be enabled, and if it's not, then a value of -1 is included, I have changed it to a binary variable such that
    #0 = hovering is not enabled and 1=hovertime enabled and used. This is done on line 276 below.

    lang_cont = additional_features[['language1', 'language2', 'language3', 'country1', 'country2', 'country3', 'uuid']]
    lang1_dum = pd.get_dummies(lang_cont['language1'])
    lang2_dum = pd.get_dummies(lang_cont['language2'])
    lang3_dum = pd.get_dummies(lang_cont['language3'])
    cont1_dum = pd.get_dummies(lang_cont['country1'])
    cont2_dum = pd.get_dummies(lang_cont['country2'])
    cont3_dum = pd.get_dummies(lang_cont['country3'])



    def rename_language_dummies(dummies, item_no, df):
        dummy_col_names = []
        for i in dummies.columns:
            col_name = i+":L"+str(item_no)
            dummy_col_names.append(col_name)
        dummies.columns = dummy_col_names
        df[dummy_col_names] = dummies
        return df.head()

    def rename_country_dummies(dummies, item_no, df):
        dummy_col_names = []
        for i in dummies.columns:
            col_name = i+":C"+str(item_no)
            dummy_col_names.append(col_name)
        dummies.columns = dummy_col_names
        df[dummy_col_names] = dummies
        return df.head()

    rename_language_dummies(lang1_dum, 1, lang_cont)
    rename_language_dummies(lang2_dum, 2, lang_cont)
    rename_language_dummies(lang3_dum, 3, lang_cont)

    rename_country_dummies(cont1_dum, 1, lang_cont)
    rename_country_dummies(cont2_dum, 2, lang_cont)
    rename_country_dummies(cont3_dum, 3, lang_cont)

    lang_cont['NumLang'] = lang_cont.iloc[:,7:101].sum(axis = 1)
    lang_cont['NumCont'] = lang_cont.iloc[:,101:236].sum(axis = 1)

    NumLangs = pd.DataFrame(lang_cont[['NumLang','uuid']])
    NumLangs = NumLangs.groupby('uuid').max()['NumLang']
    NumLangs = pd.DataFrame(NumLangs)
    NumLangs['uuid'] = NumLangs.index
    #Because going from 1 language to 2 languages isn't the same as going from 2 languages to 3 and so on, and because the vast majority
    #of users only have 1 language set, I've chosen to binarize these where 0 = 1 language and 1 = more than 1 language.


    NumConts = pd.DataFrame(lang_cont[['NumCont','uuid']])
    NumConts = NumConts.groupby('uuid').max()['NumCont']
    NumConts = pd.DataFrame(NumConts)
    NumConts['uuid'] = NumConts.index
    #Because going from 1 country to 2 countries isn't the same as going from 2 countries to 3 and so on, and because the vast majority
    #of users only have 1 country set, I've chosen to binarize these where 0 = 1 country and 1 = more than 1 country.

    get_last = pd.merge(checkins, last_checkins, how = "left", left_on='uuid', right_on='uuid')
    get_last['islast'] = get_last['time'] == get_last['last_checkin']
    get_last = get_last[get_last['islast'] == True]

    get_last_lang = get_last[['uuid','language1']]
    get_last_lang.columns = ['uuid', 'Language']
    get_last_lang['Language'] = get_last_lang['Language'].fillna('NONE')
    get_last_lang = get_last_lang.drop_duplicates()

    get_last_cont = get_last[['uuid','country1']]
    get_last_cont.columns = ['uuid', 'Country']
    get_last_cont['Country'] = get_last_cont['Country'].fillna('NONE')
    get_last_cont = get_last_cont.drop_duplicates()


    master_data = pd.merge(master_data, average_hover_time, how = 'left', left_on='uuid', right_on='uuid')
    master_data = pd.merge(master_data, browser_OS, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data = pd.merge(master_data, ulink_features, how = 'left', left_on = 'uuid', right_on = 'uuid')
    # master_data = pd.merge(master_data, pages, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data = pd.merge(master_data, NumLangs, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data['NumLang'] = (master_data['NumLang'] > 1).astype(int)
    master_data = pd.merge(master_data, NumConts, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data['NumCont'] = (master_data['NumCont'] > 1).astype(int)
    master_data = pd.merge(master_data, get_last_lang, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data = pd.merge(master_data, get_last_cont, how = 'left', left_on = 'uuid', right_on = 'uuid')
    master_data['hoverTime'] = (master_data['hoverTime'] >= 1).astype(int)
    df = master_data.drop(['registration', 'reg_date', 'num', 'last_checkin', 'time_active', 'time_active_in_sec', 'last_checkin_date_only', 'Language', 'Country' ], axis = 1)
    master_data['days_since_last_checkin'] = master_data['days_since_last_checkin'].astype(int)
    master_data['browser_count'] = master_data['Chrome']+master_data['Firefox'] + master_data['Opera'] + master_data['Opera Next'] + master_data['Safari']
    survival_data = master_data.copy()
    master_data = master_data[master_data['age_in_days'] >= 30]
    return master_data

#The feature_engineering function combines all of the functions above to make it simpler to call.
#Additionally, the data tables that are created along the way are named different things in order to be able to inspect
#them along the way for mistakes.
def feature_engineering():
    checkins, clicks = load_data()
    checkins, clicks = convert_timestamps(checkins, clicks)
    registrations, last_checkins = create_regs_last_checkins(checkins, clicks)
    master_data, churns, click_uuids, checkin_uuids = create_master_data(registrations, last_checkins, checkins, clicks)
    master_data2= add_additional_features_to_master_data(checkins, master_data, last_checkins)
    return master_data2, clicks, checkins

#A separate df is created with this function. The reason for this is that I wanted to maintain my fundamental data in one table
#(u-data), and the data that I use for my modeling in a separate table (udf). This get_df function extracts the modeling
#data table from the u_data dataframe.
def get_df(u_data):
    u_data['anti_churn'] = 1 - u_data['churn']
    udf = u_data.drop(['time_active_in_days', 'age_in_days','time_active<1hour', 'time_active<1day', 'uuid', 'Language', 'Country', 'registration','reg_date', 'num', 'last_checkin', 'last_checkin_date', 'time_active_in_sec', 'time_active', 'last_checkin_date_only', 'age', 'age_in_hours', 'days_since_last_checkin', 'churn', 'inactive1', 'inactive1-7','inactive7-15','inactive15-30'], axis = 1)
    return udf

def run_lasso(df):
    df1 = df.copy()
    y = df1.pop('anti_churn')
    X = df1
    X['browser_count'] = scale(X['browser_count'])
    y = y.astype(float)
    X = X.astype(float)
    k = X.shape[1]
    alphas = np.logspace(1, 10, 1000)
    params = 1 + np.zeros((len(alphas), k))
    for i,a in enumerate(alphas):
        X_data = scale(X)
        fit = Lasso(alpha=a, normalize=False).fit(X_data, y)
        params[i] = fit.coef_
    print "Coefficients: ", params
    fig = plt.figure(figsize=(14,6))
    for param in params.T:
        plt.plot(alphas, param)
    plt.show()

def run_ridge(df):
    df1 = df.copy()
    y = df1.pop('anti_churn')
    X = df1
    Xcols = X.columns
    X['browser_count'] = scale(X['browser_count'])
    y = y.astype(float)
    X = X.astype(float)
    k = X.shape[1]
    coefs = Ridge(normalize = True).fit(X,y).coef_
    g = 0
    for coef in coefs:
        print Xcols[g],' ', coef
        g+=1
    alphas = np.logspace(-2, 2)
    params = np.zeros((len(alphas), k))
    for i,a in enumerate(alphas):
        fit = Ridge(alpha=a, normalize=True).fit(X, y)
        params[i] = fit.coef_

    fig = plt.figure(figsize=(14,6))
    # print "Coefficients: ", Ridge(alpha=1, normalize=True).fit(X, y).coef_
    for param in params.T:
        plt.plot(alphas, param)
    plt.legend()
    plt.show()

def SMT(df, target):
    df1 = df.copy()
    y = df1.pop('anti_churn')
    X = df1
    Xcols = df1.columns
    sm = SMOTE(kind='regular', ratio = target)
    X_resampled, y_resampled = sm.fit_sample(X, y)
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    X_resampled.columns = Xcols
    y_resampled.columns = ['anti_churn']
    return X_resampled, y_resampled

def ada_boost(X,y, nf = 2, ne = 50, lr=1):
    y = y.astype(float)
    Xs = X.astype(float)
    col_names = X.columns
    Xs_t, Xs_holdout, y_t, y_holdout = train_test_split(Xs, y, train_size=.8)
    Xs_t = Xs_t.set_index([range(len(Xs_t))])
    Xs_holdout = Xs_holdout.set_index([range(len(Xs_holdout))])
    y_t = pd.DataFrame(y_t).set_index([range(len(y_t))])
    y_holdout = pd.DataFrame(y_holdout).set_index([range(len(y_holdout))])

    kf = KFold(len(Xs_t), nf)

    output_table = []
    precisions = []
    accuracies = []
    F1s = []
    fold_count = 1
    for train_index, test_index in kf:
        results = []
        Xs_train, Xs_test = Xs_t.iloc[train_index,:], Xs_t.iloc[test_index,:]
        y_train, y_test = y_t.iloc[train_index,:], y_t.iloc[test_index,:]
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        my_ada = AdaBoostClassifier(n_estimators=ne, learning_rate = lr)
        my_ada.fit(Xs_train, y_train)
        pred = my_ada.predict(Xs_test)
        pred = np.array(pred)
        output_table.append(' ')
        output_table.append("Fold "+ str(fold_count) + ':')
        output_table.append("Precision Score: "+str(precision_score(pred, y_test)))
        output_table.append("Accuracy Score: "+ str(accuracy_score(pred, y_test)))
        output_table.append("F1 Score: "+str(f1_score(pred, y_test)))
        precisions.append(precision_score(pred, y_test))
        accuracies.append(accuracy_score(pred, y_test))
        F1s.append(f1_score(pred, y_test))
        fold_count += 1
    pred_holdout = my_ada.predict(Xs_holdout)
    pred_holdout = np.array(pred_holdout)
    cm = confusion_matrix(y_holdout, pred_holdout)
    TN = cm[0][0]
    FN = cm[0][1]
    TP = cm[1][1]
    FP = cm[1][0]
    print "Mean Precision: ", np.mean(precisions)
    print "Mean F1s: ", np.mean(F1s)
    print "True Positive Rate (Sensitivity): ", TP*1./(TP+FN)#cm[1][1]*1./(cm[1][1]+cm[0][1])
    print "True Negative Rate (Specificity): ", TN*1./(TN+FP)#cm[0][0]*1./(cm[0][0]+cm[1][0])
    print "Precision: ", TP*1./(TP+FP), #precision_score(pred_holdout, y_holdout)
    print "Accuracy: ", (TP+TN)*1./(TP+TN+FP+FN), #accuracy_score(pred_holdout, y_holdout)
    indices = np.argsort(my_ada.feature_importances_)
    figure = plt.figure(figsize=(10,7))
    plt.barh(np.arange(len(col_names)), my_ada.feature_importances_[indices],
             align='center', alpha=.5)
    plt.yticks(np.arange(len(col_names)), np.array(col_names)[indices], fontsize=14)
    plt.xticks(fontsize=14)
    _ = plt.xlabel('Relative importance', fontsize=18)
    return my_ada

def random_forest(X,y, nf = 2, ne = 100, mf = 5):
    col_names = X.columns
    y = y.astype(float)
    Xs = X.astype(float)
    Xs_t, Xs_holdout, y_t, y_holdout = train_test_split(Xs, y, train_size=.8)
    Xs_t = Xs_t.set_index([range(len(Xs_t))])
    Xs_holdout = Xs_holdout.set_index([range(len(Xs_holdout))])
    y_t = pd.DataFrame(y_t).set_index([range(len(y_t))])
    y_holdout = pd.DataFrame(y_holdout).set_index([range(len(y_holdout))])

    kf = KFold(len(Xs_t), nf)

    output_table = []
    precisions = []
    accuracies = []
    F1s = []
    fold_count = 1
    for train_index, test_index in kf:
        results = []
        Xs_train, Xs_test = Xs_t.iloc[train_index,:], Xs_t.iloc[test_index,:]
        y_train, y_test = y_t.iloc[train_index,:], y_t.iloc[test_index,:]
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        rf = RandomForestClassifier(n_estimators=ne, max_features = nf)
        rf.fit(Xs_train, y_train)
        pred = rf.predict(Xs_test)
        pred = np.array(pred)
        pred = pred.round()
        output_table.append(' ')
        output_table.append("Fold "+ str(fold_count) + ':')
        output_table.append("Precision Score: "+str(precision_score(pred, y_test)))
        output_table.append("Accuracy Score: "+ str(accuracy_score(pred, y_test)))
        output_table.append("F1 Score: "+str(f1_score(pred, y_test)))
        precisions.append(precision_score(pred, y_test))
        accuracies.append(accuracy_score(pred, y_test))
        F1s.append(f1_score(pred, y_test))
        fold_count += 1
    pred_holdout = rf.predict(Xs_holdout)
    pred_holdout = np.array(pred_holdout)
    pred_holdout = pred_holdout.round()
    cm = confusion_matrix(y_holdout, pred_holdout)
    TN = cm[0][0]
    FN = cm[0][1]
    TP = cm[1][1]
    FP = cm[1][0]
    print "Mean Precision: ", np.mean(precisions)
    print "Mean F1s: ", np.mean(F1s)
    print "True Positive Rate (Sensitivity): ", TP*1./(TP+FN)#cm[1][1]*1./(cm[1][1]+cm[0][1])
    print "True Negative Rate (Specificity): ", TN*1./(TN+FP)#cm[0][0]*1./(cm[0][0]+cm[1][0])
    print "Precision: ", TP*1./(TP+FP), #precision_score(pred_holdout, y_holdout)
    print "Accuracy: ", (TP+TN)*1./(TP+TN+FP+FN), #accuracy_score(pred_holdout, y_holdout)
    indices = np.argsort(rf.feature_importances_)
    figure = plt.figure(figsize=(10,7))
    plt.barh(np.arange(len(col_names)), rf.feature_importances_[indices],
             align='center', alpha=.5)
    plt.yticks(np.arange(len(col_names)), np.array(col_names)[indices], fontsize=14)
    plt.xticks(fontsize=14)
    _ = plt.xlabel('Relative importance', fontsize=18)
    return rf

def gradient_boosting(X,y, nf = 2, lr = .1, ne = 100):
    col_names = X.columns
    y = y.astype(float)
    Xs = X.astype(float)
    Xs_t, Xs_holdout, y_t, y_holdout = train_test_split(Xs, y, train_size=.8)
    Xs_t = Xs_t.set_index([range(len(Xs_t))])
    Xs_holdout = Xs_holdout.set_index([range(len(Xs_holdout))])
    y_t = pd.DataFrame(y_t).set_index([range(len(y_t))])
    y_holdout = pd.DataFrame(y_holdout).set_index([range(len(y_holdout))])

    kf = KFold(len(Xs_t), nf)

    output_table = []
    precisions = []
    accuracies = []
    F1s = []
    fold_count = 1
    for train_index, test_index in kf:
        results = []
        Xs_train, Xs_test = Xs_t.iloc[train_index,:], Xs_t.iloc[test_index,:]
        y_train, y_test = y_t.iloc[train_index,:], y_t.iloc[test_index,:]
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        Gboost = GradientBoostingRegressor(learning_rate=lr, loss='ls', n_estimators=ne)
        Gboost.fit(Xs_train, y_train)
        pred = Gboost.predict(Xs_test)
        pred = np.array(pred)
        pred = pred.round()
        output_table.append(' ')
        output_table.append("Fold "+ str(fold_count) + ':')
        output_table.append("Precision Score: "+str(precision_score(pred, y_test)))
        output_table.append("Accuracy Score: "+ str(accuracy_score(pred, y_test)))
        output_table.append("F1 Score: "+str(f1_score(pred, y_test)))
        precisions.append(precision_score(pred, y_test))
        accuracies.append(accuracy_score(pred, y_test))
        F1s.append(f1_score(pred, y_test))
        fold_count += 1
    pred_holdout = Gboost.predict(Xs_holdout)
    pred_holdout = np.array(pred_holdout)
    pred_holdout = pred_holdout.round()
    cm = confusion_matrix(y_holdout, pred_holdout)
    TN = cm[0][0]
    FN = cm[0][1]
    TP = cm[1][1]
    FP = cm[1][0]
    print "Mean Precision: ", np.mean(precisions)
    print "Mean F1s: ", np.mean(F1s)
    print "True Positive Rate (Sensitivity): ", TP*1./(TP+FN)#cm[1][1]*1./(cm[1][1]+cm[0][1])
    print "True Negative Rate (Specificity): ", TN*1./(TN+FP)#cm[0][0]*1./(cm[0][0]+cm[1][0])
    print "Precision: ", TP*1./(TP+FP), #precision_score(pred_holdout, y_holdout)
    print "Accuracy: ", (TP+TN)*1./(TP+TN+FP+FN), #accuracy_score(pred_holdout, y_holdout)
    indices = np.argsort(Gboost.feature_importances_)
    figure = plt.figure(figsize=(10,7))
    plt.barh(np.arange(len(col_names)), Gboost.feature_importances_[indices],
             align='center', alpha=.5)
    plt.yticks(np.arange(len(col_names)), np.array(col_names)[indices], fontsize=14)
    plt.xticks(fontsize=14)
    _ = plt.xlabel('Relative importance', fontsize=18)
    return Gboost

def ada_Grid(X,y, ne = [10,50,100,200,300,500,700,800,900,1000], lr = [.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]):
    k = y.shape[0]
    parameter_grid = {'n_estimators':ne, 'learning_rate':lr}
    adaB = AdaBoostClassifier()
    y = np.array(y).reshape(k,)
    g = GridSearchCV(estimator = adaB, param_grid = parameter_grid, cv=3, scoring='precision').fit(X,y)
    print "Best score: ", g.best_score_
    print "Best parameters: ", g.best_params_
    return g.best_params_

def rf_Grid(X,y,ne=[10,20,50,100,200,300,400,500,750,1000], mf=[2,4,8,12,16,20,24]):
    k = y.shape[0]
    y = y.astype(float)
    y = np.array(y).reshape(k,)
    Xs = X.astype(float)
    parameter_grid = {'n_estimators':ne, 'max_features':mf}
    rf = RandomForestClassifier()
    g = GridSearchCV(estimator = rf, param_grid = parameter_grid, cv=3, scoring='precision').fit(X,y)
    print "Best score: ", g.best_score_
    print "Best parameters: ", g.best_params_
    return g.best_params_

def get_survival_data(u_data):
    u_data['anti_churn'] = 1 - u_data['churn']
    udf = u_data.drop(['time_active_in_days', 'time_active<1hour', 'time_active<1day', 'uuid', 'Language', 'Country', 'registration','reg_date', 'num', 'last_checkin', 'last_checkin_date', 'time_active_in_sec', 'time_active', 'last_checkin_date_only', 'age', 'age_in_hours', 'days_since_last_checkin', 'churn', 'inactive1', 'inactive1-7','inactive7-15','inactive15-30'], axis = 1)
    return udf

def gb_Grid(X,y, nf = 3, lr = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1], ne = [10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]):
    col_names = X.columns
    y = y.astype(float)
    Xs = X.astype(float)
    Xs_t, Xs_holdout, y_t, y_holdout = train_test_split(Xs, y, train_size=.8)
    Xs_t = Xs_t.set_index([range(len(Xs_t))])
    Xs_holdout = Xs_holdout.set_index([range(len(Xs_holdout))])
    y_t = pd.DataFrame(y_t).set_index([range(len(y_t))])
    y_holdout = pd.DataFrame(y_holdout).set_index([range(len(y_holdout))])

    kf = KFold(len(Xs_t), nf)

    mean_dct = {}
    final_dct = {}
    final_list = []
    output_table = []
    precisions = []
    accuracies = []
    F1s = []
    fold_count = 1
    for learn_rate in lr:
        for num_est in ne:
            for train_index, test_index in kf:
                results = []
                Xs_train, Xs_test = Xs_t.iloc[train_index,:], Xs_t.iloc[test_index,:]
                y_train, y_test = y_t.iloc[train_index,:], y_t.iloc[test_index,:]
                y_train = np.array(y_train)
                y_test = np.array(y_test)
                Gboost = GradientBoostingRegressor(learning_rate=learn_rate, loss='ls', n_estimators=num_est)
                Gboost.fit(Xs_train, y_train)
                pred = Gboost.predict(Xs_test)
                pred = np.array(pred)
                pred = pred.round()
                output_table.append(' ')
                output_table.append("Fold "+ str(fold_count) + ':')
                output_table.append("Precision Score: "+str(precision_score(pred, y_test)))
                output_table.append("Accuracy Score: "+ str(accuracy_score(pred, y_test)))
                output_table.append("F1 Score: "+str(f1_score(pred, y_test)))
                precisions.append(precision_score(pred, y_test))
                accuracies.append(accuracy_score(pred, y_test))
                F1s.append(f1_score(pred, y_test))
                fold_count += 1
            pred_holdout = Gboost.predict(Xs_holdout)
            pred_holdout = np.array(pred_holdout)
            pred_holdout = pred_holdout.round()
            cm = confusion_matrix(y_holdout, pred_holdout)
            TN = cm[0][0]
            FN = cm[0][1]
            TP = cm[1][1]
            FP = cm[1][0]
            final_list.append("LR: " + str(learn_rate) + "NE: " + str(num_est) + "Mean Precision: " + str(np.mean(precisions)) + "True Positive Rate (Sensitivity): " + str(TP*1./(TP+FN)) + "Precision: " + str(TP*1./(TP+FP)))
            ref = "LR: "+str(learn_rate)+" " +"NE: "+str(num_est)
            mean_dct[ref] = np.mean(precisions)
            final_dct[ref] = TP*1./(TP+FP)
            # print "Mean Precision: ", np.mean(precisions)
            # print "Mean F1s: ", np.mean(F1s)
            # print "True Positive Rate (Sensitivity): ", TP*1./(TP+FN)#cm[1][1]*1./(cm[1][1]+cm[0][1])
            # print "True Negative Rate (Specificity): ", TN*1./(TN+FP)#cm[0][0]*1./(cm[0][0]+cm[1][0])
            # print "Precision: ", TP*1./(TP+FP), #precision_score(pred_holdout, y_holdout)
            # print "Accuracy: ", (TP+TN)*1./(TP+TN+FP+FN), #accuracy_score(pred_holdout, y_holdout)

    return mean_dct, final_dct, final_list


if __name__ == '__main__':
    u_data, uclicks, ucheckins = feature_engineering()
    udf = get_df(u_data)
    X10, y10 = SMT(udf, .1112)
    X20, y20 = SMT(udf, .25001)
    X30, y30 = SMT(udf, .42861)
    X40, y40 = SMT(udf, .6667)
    X50, y50 = SMT(udf, .99999999999999999)
    survival_data = get_survival_data(u_data)
    # ada_Grid(X10,y10)
    # ada_Grid(X20,y20)
    # ada_Grid(X30,y30)
    # ada_Grid(X40,y40)
    # ada_Grid(X50,y50)
    # rf_Grid(X10,y10)
    # rf_Grid(X20,y20)
    # rf_Grid(X30,y30)
    # rf_Grid(X40,y40)
    # rf_Grid(X50,y50)

    #ada_boost(X10, y10, nf = 3, ne = 450, lr=1.3)
    # Best score:  0.644242996538
    # Best parameters:  {'n_estimators': 100, 'learning_rate': 1.6}

    # ada_Grid(X20,y20, ne = [2,4,6,8,10,12], lr = [1.6,1.7,1.8,1.9,2])
    # #Best score:  0.825614636935
    # #Best parameters:  {'n_estimators': 6, 'learning_rate': 1.9}

    # ada_Grid(X30,y30, ne = [900,950,1000,1100,1200], lr = [1.5,1.6,1.7])
    # #Best score:  0.669836867436
    # #Best parameters:  {'n_estimators': 900, 'learning_rate': 1.5}

    # ada_Grid(X40,y40, ne = [250,300,350,400,500], lr = [1.7,1.8,1.9,2])
    # #Best score:  0.754996416638
    # #Best parameters:  {'n_estimators': 250, 'learning_rate': 1.9}

    # ada_Grid(X50,y50, ne = [400,500,600,700], lr = [1.7,1.8,1.9,2])
    # #Best score:  0.70155675008
    # #Best parameters:  {'n_estimators': 700, 'learning_rate': 1.8}

    rf_Grid(X10,y10,ne=[150,200,250,400], mf=[1,2,3,4])
    # Best score:  0.834973305731
    # Best parameters:  {'max_features': 3, 'n_estimators': 250}

    # rf_Grid(X20,y20,ne=[250,300,350,400], mf=[1,2,3])
    # #Best score:  0.654511774491
    # #Best parameters:  {'max_features': 2, 'n_estimators': 250}

    # rf_Grid(X30,y30,ne=[5,7,9,10,12,14,16], mf=[2,3,4,5,6])
    # # Best score:  0.685458338107
    # # Best parameters:  {'max_features': 2, 'n_estimators': 12}

    rf_Grid(X40,y40,ne=[5,7,9,10,12,14,15], mf=[10,11,12,13,14])
    # Best score:  0.752268550875
    # Best parameters:  {'max_features': 12, 'n_estimators': 14}

    # rf_Grid(X50,y50,ne=[40,45,50,55,60,70,75], mf=[1,2,3,4])
    # # Best score:  0.658360724325
    # # Best parameters:  {'max_features': 1, 'n_estimators': 70}

    mean_dct10, final_dct10, final_list10 = gb_Grid(X10,y10)
    mean_dct20, final_dct20, final_list20 = gb_Grid(X20,y20)
    mean_dct30, final_dct30, final_list30 = gb_Grid(X30,y30)
    mean_dct40, final_dct40, final_list40 = gb_Grid(X40,y40)
    mean_dct50, final_dct50, final_list50 = gb_Grid(X50,y50)

# uuid_counts_by_ip = checkins.groupby('ip').count()['uuid']
# uuid_counts_by_ip = pd.DataFrame(uuid_counts_by_ip)
# uuid_counts_by_ip['IP'] = uuid_counts_by_ip.index
# #one_uuid = uuid_counts_by_ip == 1
# #one_uuid = set(one_uuid)
# m1 = pd.merge(master_data, uuid_counts_by_ip, how = 'left', left_on = 'ip', right_on = 'IP')
#
#
# ip_counts_by_uuid = checkins.groupby('uuid').count()['ip']
# ip_counts_by_uuid = pd.DataFrame(ip_counts_by_uuid)
# ip_counts_by_uuid['UUID'] = ip_counts_by_uuid.index
# mu = pd.merge(master_data, ip_counts_by_uuid, how = 'left', left_on = 'ip', right_on='ip')
