from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVR, LinearSVC, SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost.sklearn import XGBRegressor, XGBClassifier
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

from sklearn.metrics import mean_absolute_error, f1_score, log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=DeprecationWarning)

# cuml 即使设置了 random_state 目前也无法保证完全一致的结果
from cuml.ensemble import RandomForestClassifier as cuRFC, RandomForestRegressor as cuRFR
from cuml import LogisticRegression as cuLR, Lasso as cuLasso
from imblearn.over_sampling import SMOTE


def one_relative_abs_func(greater_is_better=True):
    def one_relative_abs(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        one_mae = 1 - mae / np.mean(np.abs(y_true - np.mean(y_true)))
        # print(one_mae,np.abs(one_mae))
        return np.abs(one_mae)

    scorefunc = make_scorer(one_relative_abs, greater_is_better=greater_is_better)
    return scorefunc


def one_relative_abs(y_true,y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    one_mae = 1 - mae/np.mean(np.abs(y_true - np.mean(y_true)))
    #print(one_mae,np.abs(one_mae))
    return np.abs(one_mae)


def one_rmse_func(greater_is_better=False):
    def rmse_func(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true,y_pred))
        # print(one_mae,np.abs(one_mae))
        return rmse

    scorefunc = make_scorer(rmse_func, greater_is_better=greater_is_better)
    return scorefunc


class Evaluater(object):
    """docstring for Evaluater"""

    def __init__(self, cv=5, stratified=True, n_jobs=1, tasktype="C", evaluatertype="rf", n_estimators=5,
                 random_state=np.random.randint(100000), greater_is_better=False, balance_data=False, use_cuml=False,
                 default_param=False, **kargs):
        # tasktype = "C" or "R" for classification or regression
        # evaluatertype = 'rf', 'svm', 'lr' for random forest, SVM, logisticregression
        self.random_state = random_state
        self.cv = cv
        self.stratified = stratified
        self.greater_is_better = greater_is_better
        self.n_jobs = n_jobs
        self.tasktype = tasktype
        self.balance_data = balance_data
        self.use_cuml = use_cuml
        if self.tasktype == "C":
            self.kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)  # 分层抽样
        else:
            self.kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)

        if evaluatertype == 'rf':
            if tasktype == "C":
                if self.use_cuml:
                    self.clf = cuRFC(n_estimators=n_estimators, random_state=self.random_state)
                else:
                    if default_param:
                        self.clf = RandomForestClassifier(random_state=self.random_state, n_jobs=n_jobs)
                    else:
                        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                                          random_state=self.random_state, n_jobs=n_jobs)
            elif tasktype == "R":
                if use_cuml:
                    self.clf = cuRFR(n_estimators=10, random_state=self.random_state, **kargs)
                else:
                    if default_param:
                        self.clf = RandomForestRegressor(n_jobs=n_jobs, random_state=self.random_state)
                    else:
                        self.clf = RandomForestRegressor(n_estimators=n_estimators,
                                                         random_state=self.random_state, n_jobs=n_jobs)
        elif evaluatertype == "lr":
            if tasktype == "C":
                if use_cuml:
                    self.clf = cuLR()
                else:
                    self.clf = LogisticRegression(random_state=self.random_state)
            elif tasktype == "R":
                if use_cuml:
                    self.clf = cuLasso(random_state=self.random_state)
                else:
                    self.clf = Lasso(random_state=self.random_state)
        elif evaluatertype == 'dt':
            if tasktype == 'C':
                self.clf = DecisionTreeClassifier(random_state=self.random_state)
            else:
                self.clf = DecisionTreeRegressor(random_state=self.random_state)
        elif evaluatertype == 'linear_svm':
            if tasktype == 'C':
                self.clf = LinearSVC(random_state=self.random_state)
            else:
                self.clf = LinearSVR(random_state=self.random_state)
        elif evaluatertype == 'svm':
            if tasktype == 'C':
                self.clf = SVC(random_state=self.random_state)
            else:
                self.clf = SVR()
        elif evaluatertype == 'xgboost':
            if tasktype == 'C':
                self.clf = XGBClassifier(random_state=self.random_state)
            else:
                self.clf = XGBRegressor(random_state=self.random_state)
        elif evaluatertype == 'lightgbm':
            if tasktype == 'C':
                self.clf = LGBMClassifier(random_state=self.random_state)
            else:
                self.clf = LGBMRegressor(random_state=self.random_state)


    # @profile
    def CV(self, X, y):
        if y.shape[1] == 1:
            y = y[:, 0]
        X = np.nan_to_num(X)
        X = np.clip(X, -3e38, 3e38)
        if self.greater_is_better:
            scoring = 'f1' if self.tasktype == "C" else one_relative_abs_func()
        else:
            scoring = 'f1' if self.tasktype == "C" else one_rmse_func()

        score = cross_val_score(self.clf, X, y, scoring=scoring, cv=self.kf, n_jobs=self.n_jobs)

        # print(X,y)
        # print(X.shape)
        # print("cv score",score,score.mean())
        return abs(score.mean())

    def CV2(self, X, y):
        if y.shape[1] == 1:
            y = y[:, 0]
        res = []

        if self.use_cuml:
            # X = cp.asarray(X.astype(np.float32))
            # y = cp.asarray(y.astype(np.float32))
            X = X.astype(np.float32)
            y = y.astype(np.float32)

        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.clf.fit(X_train, y_train)
            y_test_hat = self.clf.predict(X_test)
            # if self.use_cuml:
            #     y_test = y_test.get()
            #     y_test_hat = y_test_hat.get()
            res.append(self.metrics(y_test, y_test_hat))

        return np.array(res).mean(axis=0)

    def CV2_test(self, X, y):
        if y.shape[1] == 1:
            y = y[:, 0]
        rmse = []
        mse = []
        mae = []
        r2 = []

        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.clf.fit(X_train, y_train)
            y_test_hat = self.clf.predict(X_test)
            # if self.use_cuml:
            #     y_test = y_test.get()
            #     y_test_hat = y_test_hat.get()
            rmse.append(np.sqrt(mean_squared_error(y_test, y_test_hat)))
            mse.append(mean_squared_error(y_test, y_test_hat))
            mae.append(mean_absolute_error(y_test, y_test_hat))
            r2.append(r2_score(y_test, y_test_hat))

        return np.array(rmse).mean(axis=0), np.array(mse).mean(axis=0), np.array(mae).mean(axis=0), np.array(r2).mean(axis=0)


    def metrics(self, y_true, y_pred):
        if self.tasktype == "C":
            f_score = f1_score(y_true, y_pred, average='micro')
            # f_score = f1_score(y_true, y_pred, average='macro')
            # auc = roc_auc_score(y_true, y_pred)
            # logloss = log_loss(y_true, y_pred)
            # return f_score, auc, logloss
            return f_score
        else:
            # rel_MAE = 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true - np.mean(y_true)))
            # rel_MSE = 1 - mean_squared_error(y_true, y_pred) / np.mean(np.square((y_true - np.mean(y_true))))
            # print(mean_absolute_error(y_true,y_pred))

            if self.greater_is_better:
                res = 1 - mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true - np.mean(y_true)))
            else:
                res = np.sqrt(mean_squared_error(y_true, y_pred))
            return res
