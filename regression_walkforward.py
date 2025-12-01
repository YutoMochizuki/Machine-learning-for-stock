'''
regression script (linear, ransac, ridge, lasso, elasticnet, svs, random forest)
'''

import datetime
import glob
import io
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import constants as const

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterSampler


plt.rcParams['font.size'] = 10
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams['font.family'] ='Times New Roman'
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.framealpha"] = 1.0
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.fancybox"] = False
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.grid"] = False

sns.set_style("ticks", {
    'axes.facecolor': "1.0",
    'xtick.major.size': 10.0,
    'ytick.major.size': 10.0,
    'xtick.minor.size': 6.0,
    'ytick.minor.size': 6.0,
    'xtick.direction': u'in',
    'ytick.direction': u'in',
    })
sns.set_context("talk", 1.0)
font = {"family":"Noto Sans CJK JP"}
mpl.rc('font', **font)
palette = sns.color_palette()


random_num = 0


def pickl_to_pd(table_list):
    '''
    pandas pikle
    '''
    df = pd.concat([pd.read_pickle(x) for x in table_list])
    return df


def txt_to_pandas(file_path, columns):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            #  1行目をスキップ
            if line.startswith('#'):
                continue
            values = line.strip().split()  # スペースで分割
            data.append(values)
    # データフレームを作成
    df = pd.DataFrame(data, columns=columns)
    return df


def preprocess(df):
    # 欠損値処理
    if df.isnull().values.any():
        # 行を削除
        df = df.dropna(how='any')

        # 平均値で補完
        # imputer = SimpleImputer(strategy='mean')
        # df = imputer.fit_transform(df)

        df = df.reset_index(drop=True)

    df = pd.get_dummies(df, dtype=int)
    df = df.astype(float)
    return df


def training_normalization(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_num)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # 平均、分散
    print('mean: ', sc.mean_)
    print('var: ', sc.var_)

    return X_train_std, X_test_std, y_train, y_test


def linear_regression(X_train_std, X_test_std, y_train, y_test,explained_variance, objected_variance, objected_variance_index):
    slr = LinearRegression()
    slr.fit(X_train_std, y_train)
    y_train_pred = slr.predict(X_train_std)
    y_test_pred = slr.predict(X_test_std)

    print('回帰係数：', slr.coef_[0])
    print('切片：', slr.intercept_)
    print('決定係数：', slr.score(X_test_std, y_test))

    plot(X_test_std, y_test_pred, y_test_pred, explained_variance, objected_variance)

    return y_train_pred, y_test_pred


def ransac_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, residual_threshold=5.0, random_state=random_num, loss='absolute_error')
    ransac.fit(X_train_std, y_train)
    y_train_pred = ransac.predict(X_train_std)
    y_test_pred = ransac.predict(X_test_std)

     # 正常値をtrue
    inlier_mask = ransac.inlier_mask_
    # 外れ値をtrue
    outlier_mask = np.logical_not(inlier_mask)

    print('回帰係数：', ransac.estimator_.coef_[0])
    print('切片：', ransac.estimator_.intercept_)
    print('決定係数：', ransac.score(X_test_std, y_test))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)

    return y_train_pred, y_test_pred


def ridge_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):

    param_dist = {
    "alpha": np.logspace(-1,3,100),
    "max_iter": [i for i in range(int(1e3), int(1e4), 100)]
}

    n_iter = 30 # ランダムに試すパラメータ数
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_num))

    best_params, best_score = walk_forward(param_list, X_train_std, y_train, Ridge)
    print("ウォークフォワード法ベストパラメータ:", best_params)
    print("ウォークフォワード法平均R²:", best_score)

    max_iter_ridge = best_params['max_iter']
    alpha_ridge = best_params['alpha']

    ridge = Ridge(alpha=alpha_ridge, max_iter=max_iter_ridge)
    ridge.fit(X_train_std, y_train)
    y_train_pred = ridge.predict(X_train_std)
    y_test_pred = ridge.predict(X_test_std)

    print('回帰係数：', ridge.coef_[0])
    print('切片：', ridge.intercept_)
    print('決定係数：', ridge.score(X_test_std, y_test))
    MSE = np.mean((y_test - y_test_pred) ** 2)
    print('平均二乗誤差：', MSE)
    MAE = np.mean(np.abs(y_test - y_test_pred))
    print('平均絶対誤差：', MAE)

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def lasso_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):

    param_dist = {
    "alpha": np.logspace(-1,3,100),
    "max_iter": [i for i in range(int(1e3), int(1e4), 100)]
}

    n_iter = 30 # ランダムに試すパラメータ数
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_num))

    best_params, best_score = walk_forward(param_list, X_train_std, y_train, Ridge)
    print("ウォークフォワード法ベストパラメータ:", best_params)
    print("ウォークフォワード法平均R²:", best_score)

    max_iter_lasso = best_params['max_iter']
    alpha_lasso = best_params['alpha']

    lasso = Lasso(alpha=alpha_lasso, max_iter=max_iter_lasso)
    lasso.fit(X_train_std, y_train)
    y_train_pred = lasso.predict(X_train_std)
    y_test_pred = lasso.predict(X_test_std)

    print('回帰係数：', lasso.coef_[0])
    print('切片：', lasso.intercept_)
    print('決定係数：', lasso.score(X_test_std, y_test))
    MSE = np.mean((y_test - y_test_pred) ** 2)
    print('平均二乗誤差：', MSE)
    MAE = np.mean(np.abs(y_test - y_test_pred))
    print('平均絶対誤差：', MAE)

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def elasticnet_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):

    # --- 修正1: パラメータ範囲の改善 ---
    param_dist = {
        "alpha": np.logspace(-4, 2, 100),  # 0.0001 ~ 100 まで広く探索
        "l1_ratio": np.linspace(0.1, 1.0, 10) # 0を含めると収束警告が出やすいため0.1から推奨
    }

    # max_iterは探索せず、モデル定義時に固定値として与えるのが定石です
    MAX_ITER_FIXED = 10000

    n_iter = 30
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    # --- 修正2: walk_forwardに正しいクラス(ElasticNet)を渡す ---
    # 注: walk_forward内部でモデルをインスタンス化する際、max_iter=MAX_ITER_FIXED を渡せるように
    # 実装されている必要があります。もしkwargsを受け取れないなら、ここでの指定方法は工夫が必要です。
    # ここでは一般的なsklearnのパターンとしてクラスを渡すと仮定します。

    print("Walk-forward optimization start...")
    # RidgeではなくElasticNetを渡す
    best_params, best_score = walk_forward(param_list, X_train_std, y_train, ElasticNet)

    print("ウォークフォワード法ベストパラメータ:", best_params)
    print("ウォークフォワード法平均スコア:", best_score)

    # ベストパラメータの抽出
    alpha_best = best_params['alpha']
    l1_ratio_best = best_params['l1_ratio']

    # --- モデルの再学習 ---
    # max_iterはここで十分な値を指定
    elasticnet = ElasticNet(alpha=alpha_best, l1_ratio=l1_ratio_best, max_iter=MAX_ITER_FIXED)
    elasticnet.fit(X_train_std, y_train)

    y_train_pred = elasticnet.predict(X_train_std)
    y_test_pred = elasticnet.predict(X_test_std)

    # --- 修正3: 結果表示の適正化 ---
    print('--- Result ---')
    print('切片 (Intercept):', elasticnet.intercept_)
    # 全係数を表示（ベクトルとして表示）
    print('回帰係数 (Coefficients):\n', elasticnet.coef_)

    # ゼロになった係数の数をカウント（スパース性の確認）
    n_zero = np.sum(elasticnet.coef_ == 0)
    print(f'係数が0になった変数の数: {n_zero} / {len(elasticnet.coef_)}')

    print('決定係数 (Test R2):', elasticnet.score(X_test_std, y_test))

    MSE = np.mean((y_test - y_test_pred) ** 2)
    print('平均二乗誤差 (MSE):', MSE)
    MAE = np.mean(np.abs(y_test - y_test_pred))
    print('平均絶対誤差 (MAE):', MAE)

    # プロット関数（既存のものを使用）
    # plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)

    return y_train_pred, y_test_pred


def svr_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    # 探索パラメータ
    param_dist = {
            'C': [1, 10, 100],
        'gamma': ['scale'],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    }

    n_iter = 30 # ランダムに試すパラメータ数
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_num))
    best_params, best_score = walk_forward(param_list, X_train_std, y_train, SVR)

    svr = SVR(kernel=best_params['kernel'], gamma=best_params['gamma'], C=best_params['C'])
    svr.fit(X_train_std, y_train)
    y_train_pred = svr.predict(X_train_std)
    y_test_pred = svr.predict(X_test_std)

    print('回帰係数：', svr.coef_[0])
    print('切片：', svr.intercept_)
    print('決定係数：', svr.score(X_test_std, y_test))
    print('平均二乗誤差：', np.mean((y_test - y_test_pred) ** 2))
    print('平均絶対誤差：', np.mean(np.abs(y_test - y_test_pred)))

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def random_forest(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):
    param_dist = {
    'n_estimators': [i for i in range(int(1e1), int(1e2), 10)],
    }

    n_iter = 30 # ランダムに試すパラメータ数
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

    best_params, best_score = walk_forward(param_list, X_train_std, y_train, RandomForestRegressor)
    print("ウォークフォワード法ベストパラメータ:", best_params)
    print("ウォークフォワード法平均R²:", best_score)

    max_iter_random = best_params['n_estimators']

    rf = RandomForestRegressor(n_estimators=max_iter_random, random_state=random_num, n_jobs=2)
    rf.fit(X_train_std, y_train)
    y_train_pred = rf.predict(X_train_std)
    y_test_pred = rf.predict(X_test_std)

    print('決定係数：', rf.score(X_test_std, y_test))
    MSE = np.mean((y_test - y_test_pred) ** 2)
    print('平均二乗誤差：', MSE)
    MAE = np.mean(np.abs(y_test - y_test_pred))
    print('平均絶対誤差：', MAE)

    plot(X_test_std, y_test, y_test_pred, explained_variance, objected_variance)
    return y_train_pred, y_test_pred


def nn(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index):

    from sklearn.neural_network import MLPRegressor
    param_dist_nn = {
    "max_iter": [i for i in range(int(1e3), int(1e4), 100)]
}
    n_iter_nn = 30 # ランダムに試すパラメータ数
    param_list_nn = list(ParameterSampler(param_dist_nn, n_iter=n_iter_nn, random_state=random_num))
    best_params_nn, best_score_nn = walk_forward(param_list_nn, X_train_std, y_train, MLPRegressor)
    print("ウォークフォワード法ベストパラメータ (NN):", best_params_nn)
    print("ウォークフォワード法平均R² (NN):", best_score_nn)
    max_iter_nn = best_params_nn['max_iter']
    sc_nn = StandardScaler()
    sc_nn.fit(X_train_std)
    x_scaled_train = sc_nn.transform(X_train_std)
    x_scaled_valid = sc_nn.transform(X_test_std)
    model_nn = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=max_iter_nn, random_state=random_num)
    model_nn.fit(x_scaled_train, y_train)
    y_train_pred_nn = model_nn.predict(x_scaled_train)
    y_valid_pred_nn = model_nn.predict(x_scaled_valid)

    print('決定係数 (NN):', model_nn.score(x_scaled_valid, y_test))
    print('平均二乗誤差 (NN):', np.mean((y_test - y_valid_pred_nn) ** 2))
    print('平均絶対誤差 (NN):', np.mean(np.abs(y_test - y_valid_pred_nn)))
    plot(X_test_std, y_test, y_valid_pred_nn, explained_variance, objected_variance)
    return y_train_pred_nn, y_valid_pred_nn


def cost_func(X, y, n_cluster):
    cost = 0
    for k in range(int(n_cluster)):
        cost += sum((X[y==k,0] - X[y==k,0].mean(axis=0))**2 + (X[y==k,1] - X[y==k,1].mean(axis=0))**2)
    return cost


def plot(X, y, y_pred, explained_variance, objected_variance):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    # plt.scatter(X[:, objected_variance_index], y, color='blue', label='Data')
    # plt.plot(X[:, objected_variance_index], y_pred, color='red', label='RANSAC Regression Line')
    plt.xlabel(explained_variance)
    plt.ylabel(objected_variance)
    plt.legend()
    plt.tight_layout()
    plt.savefig('regression.png')
    plt.close()


def plot_residuals(y_train, y_pred_train, y_test, y_pred_test):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train - y_pred_train, c='blue', marker='o', label='Train data')
    plt.scatter(y_test, y_test - y_pred_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Observed values')
    plt.ylabel('Residuals')
    plt.hlines(y=0, xmin=min(y_train.min(), y_test.min()), xmax=max(y_train.max(), y_test.max()), color='red')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('residuals.png')
    plt.close()


def walk_forward(param_list, x_scaled_train_full, y, method):
    # ウォークフォワード法の設定
    initial_train_size = 100  # 最初に学習するサンプル数
    step = 10                 # 1回ごとのテストサイズ

    best_score = -np.inf
    best_params = None

    # ウォークフォワード法でハイパーパラメータ評価
    for params in param_list:
        scores = []
        start = initial_train_size
        while start < len(x_scaled_train_full):
            x_train = x_scaled_train_full[:start]
            y_train_slice = y[:start]
            x_valid = x_scaled_train_full[start:start+step]
            y_valid_slice = y[start:start+step]

            # データが空でないか確認
            if len(x_train) > 0 and len(x_valid) > 0:
                model = method(**params, random_state=42)
                model.fit(x_train, y_train_slice)
                y_pred = model.predict(x_valid)
                scores.append(r2_score(y_valid_slice, y_pred))

            start += step

        # スコアが計算できた場合のみ平均を計算
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    return best_params, best_score


def main():
    # sepスペース区切り、headerなし
    df = pd.read_csv('/Users/mochizuki/Documents/study/kaggle/housing.csv', header=None, sep='\s+')
    print(df.head())
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    df = df[cols]
    df = preprocess(df)
    print(df.head())

    explained_variance = 'RM'
    objected_variance = 'MEDV'
    objected_variance_index = cols.index(objected_variance) - 1
    X = df[[explained_variance]].values
    # X = df.iloc[:, :-1].values
    y = df[objected_variance].values

    X_train_std, X_test_std, y_train, y_test = training_normalization(X, y)

    # y_pred_train, y_pred_test = linear_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = ransac_regression(X_train_std,  X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = ridge_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    y_pred_train, y_pred_test = lasso_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = elasticnet_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = svr_regression(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    # y_pred_train, y_pred_test = random_forest(X_train_std, X_test_std, y_train, y_test, explained_variance, objected_variance, objected_variance_index)

    plot_residuals(y_train, y_pred_train, y_test, y_pred_test)


if __name__ == "__main__":
    main()
