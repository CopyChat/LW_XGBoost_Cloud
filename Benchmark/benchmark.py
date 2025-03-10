"""
to apply ML method to improve cloud fraction estimation,
compared with traditional physical cloud method based on thermal physics,
where assumptions made in lower level of atmosphere.

This file is to analysis the XGBoost results, all ML works are deal with MS Code.
"""

__version__ = f'Version 1.0  \nTime-stamp: <2024-11-06>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import joblib
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr
import hydra
from omegaconf import DictConfig
import numpy as np

# import subprocess
# import numpy as np
# import pandas as pd
# import xarray as xr
# from importlib import reload

from importlib import reload
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

import GEO_PLOT
import RESEARCH
import pickle
import PUBLISH
from sklearn.model_selection import train_test_split

def split(data, tst_sz):
    y = data["CF"]
    X = data.drop("CF" , axis=1)
    # X = X.drop("timestamp" , axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=tst_sz, random_state=7)
    return X_train, X_test, y_train, y_test

# Global data:
# Global constant definition (naming in uppercase)

# ============================= read data ===========================
data_set_name = '../dataset/raw.bsrn_lacy.2019_2022.5min.local_time.csv'
# data = pd.read_csv(path_data+data_set_name,delim_whitespace = False)
df_raw = GEO_PLOT.read_csv_into_df_with_header(data_set_name)
print(df_raw.size)
df_raw.corr(method='pearson')
# ----------------------------  split data:
# works on two-year data:
# for training (SearchGrid and CV) and valid
train_valid = df_raw['2019-09-13':'2021-09-12']
X_train, X_valid, y_train, y_valid = split(train_valid, 0.1)

predictors = list(df_raw.columns)
# #predictors.remove('P')
# #predictors.remove('RH')
predictors.remove('CF')

# X_train = df_raw['2019-09-13':'2021-09-12'][predictors]
# y_train = df_raw['2019-09-13':'2021-09-12'][{'CF'}]

X_test = df_raw['2021-10-01':'2022-09-28'][predictors]
y_test = df_raw['2021-10-01':'2022-09-28'][['CF']]

print(X_train.columns)
# data for training and valid, to verify model.
# only train data is used for searching parameter with CV.
# define evalSet for monitoring train and valid process for early_stopping and overfitting.

# Fit scaler on training data only
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the fitted scaler
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

evalSet = [(X_train, y_train), (X_valid, y_valid)]

# ============================= done of read data ===========================
# functions:

def ann_model_train(X_train, X_valid, X_test, model_save:str):
    """
    build an ANN model and train
    :return:
    """
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.optimizers import Adam

    # Define the model
    model = Sequential([
        Input(shape=(7,)),  # Explicit input layer with 7 features
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=512,
        verbose=1
    )

    # Extract RMSE values
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']

    # Plot RMSE learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    # plt.xlabel('Epochs')
    # plt.ylabel('RMSE')
    plt.title('Learning Curve RMSE')
    plt.legend()
    plt.savefig(f'./ann_trained.png')
    plt.show()

    # save the tuned model
    # Save the model in .h5 format
    model.save(model_save)  # This will save both architecture and weights

    # Load the model
    loaded_model = load_model(model_save)
    # loaded_model = load_model(ann_model, custom_objects={'rmse': rmse})

    # Make predictions on X_test
    predictions = loaded_model.predict(X_test)

    return loaded_model

    scr_train = round(model.score(X_train, y_train),4)
    scr_test  = round(model.score(X_test, y_test),4)
    y_pred   = model.predict(X_test)

    resultat = model.evals_result()

    plt.figure(figsize=(8, 8))

    plt.plot(resultat['validation_0']['rmse'], label = 'train=%f' %scr_train, color='blue')
    plt.plot(resultat['validation_1']['rmse'], label = 'test=%f' %scr_test, color='red')

    plt.tick_params(labelsize=14)
    plt.ylabel('', fontsize=148)
    plt.xlabel('', fontsize=148)
    plt.title("Learning curve RMSE", fontsize=14)
    plt.legend(prop={'size':15})
    plt.savefig(fig_name + ".png")
    plt.show()

def upgrade_model(model, param_name, param_value):
    print(f'old {param_name} is: {model.get_params()[param_name]}')
    # upgrade model parameter:
    params = {param_name: param_value}
    model.set_params(**params)
    print(f'new {param_name} is: {model.get_params()[param_name]}')

def plot_learning_curve(model, evalSet, fig_name='learning_curve.png'):
    """
    using train and test valid datasets, but the showing "testing curve"
    model:
    evalSet:
    :return:
    """
    print(f'start training...')
    model.fit(X_train, y_train, verbose=False, eval_set=evalSet)

    # plotting:
    scr_train = round(model.score(X_train, y_train), 4)
    scr_test = round(model.score(X_valid, y_valid), 4)
    y_pred = model.predict(X_valid)

    resultat = model.evals_result()

    plt.figure(figsize=(8, 8))

    plt.plot(resultat['validation_0']['rmse'], label='train=%f' % scr_train, color='blue')
    plt.plot(resultat['validation_1']['rmse'], label='test=%f' % scr_test, color='red')

    plt.tick_params(labelsize=14)
    plt.ylabel('', fontsize=148)
    plt.xlabel('', fontsize=148)
    plt.title("Learning curve RMSE", fontsize=14)
    plt.legend(prop={'size': 15})
    plt.savefig(f'./{fig_name:s}')
    plt.show()


def calculate_statistics(df):
    # Ensure required columns are present
    if not {'CF', 'CF_pred'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'CF' and 'CF_pred' columns.")

    # Drop missing values to avoid errors in calculations
    df = df[['CF', 'CF_pred']].dropna()

    # Calculate RMSE
    rmse = np.sqrt(((df['CF'] - df['CF_pred']) ** 2).mean())

    # Calculate MAB (Mean Absolute Bias)
    mab = (df['CF'] - df['CF_pred']).abs().mean()

    # Calculate COR (Pearson Correlation Coefficient)
    cor, _ = pearsonr(df['CF'], df['CF_pred'])

    # Calculate MES (Mean Error Square)
    mes = ((df['CF'] - df['CF_pred']) ** 2).mean()

    # Calculate R^2 (Coefficient of Determination)
    ss_res = ((df['CF'] - df['CF_pred']) ** 2).sum()
    ss_tot = ((df['CF'] - df['CF'].mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    # Return results in a dictionary
    output = {
        'MAB': mab,
        'RMSE': rmse,
        'COR': cor,
        'R_squared': r_squared,
    }

    # Print the results with 2.2f format, aligned
    print("\nStatistics:")
    for key, value in output.items():
        print(f"{key:<20}: {value:6.2f}")

    return output


def rf_model_train(model_save: str):
    import sklearn  # Import to access version
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import learning_curve

    # Step 1: Define XGBoost parameters (for reference)
    # that is the params optimised by the code in ../XGBoost at Google Colab using GPUs
    # the CPU version may be different
    xgb_params = {
        'base_score': 0.5,
        'colsample_bytree': 0.8,
        'learning_rate': 0.2,
        'max_depth': 10,
        'min_child_weight': 50,
        'n_estimators': 200,
        'subsample': 0.9,
        'reg_alpha': 10,
        'reg_lambda': 0,
    }

    # Step 2: Create a similar Random Forest Regressor
    rf_params = {
        'n_estimators': xgb_params['n_estimators'],  # Match number of trees
        'max_depth': xgb_params['max_depth'],  # Similar depth
        'max_features': xgb_params['colsample_bytree'],  # Approximate colsample_bytree
        'min_samples_split': xgb_params['min_child_weight'],  # Approximate min_child_weight
        'random_state': 42,  # For reproducibility
    }

    rf_model = RandomForestRegressor(**rf_params)

    print(f'Train the model on X_train, y_train')
    rf_model.fit(X_train, y_train)

    print(f'Learning curve validation by RMSE')
    train_sizes, train_scores, valid_scores = learning_curve(
        rf_model, X_train, y_train, cv=10, scoring="neg_root_mean_squared_error",
        train_sizes = np.linspace(0.1, 1.0, 5), n_jobs=-1)

    train_rmse = -train_scores.mean(axis=1)
    valid_rmse = -valid_scores.mean(axis=1)

    # Plotting learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, label="Training score: ")
    plt.plot(train_sizes, valid_rmse, label="Validation RMSE")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title("Learning Curve for Random Forest Regressor")
    plt.legend()
    plt.savefig('./rf_trained.png')
    plt.show()

    # Step 5: Evaluate on X_test, y_test using RMSE
    y_test_pred = rf_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")

    # Step 6: Save the model with version information (sklearn version)
    sklearn_version = sklearn.__version__  # Retrieve sklearn version
    model_filename = f"{model_save}"
    joblib.dump((rf_model, sklearn_version), model_filename)

    # Step 7: Reload the model for further use
    loaded_model, loaded_version = joblib.load(model_filename)
    print(f"Loaded model version (scikit-learn): {loaded_version}")

    return loaded_model


def svm_model_train(X_train, X_valid, X_test, model_save: str):
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import mean_squared_error

    # 初始化SVM回归模型
    svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    print(f'training')
    svm_model.fit(X_train, y_train)

    print(f'使用learning_curve函数计算训练集和验证集的分数')
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=svm_model,
        X=X_train_scaled,
        y=y_train,
        cv=5,  # 5折交叉验证
        scoring="neg_root_mean_squared_error",  # 使用负RMSE
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)  # 训练集大小从10%到100%
    )

    # 将负RMSE转换为正RMSE
    train_rmse = -train_scores
    valid_rmse = -valid_scores

    # 计算训练和验证RMSE的均值和标准差
    train_rmse_mean = train_rmse.mean(axis=1)
    train_rmse_std = train_rmse.std(axis=1)
    valid_rmse_mean = valid_rmse.mean(axis=1)
    valid_rmse_std = valid_rmse.std(axis=1)

    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse_mean, 'o-', color="blue", label="Training RMSE")
    plt.plot(train_sizes, valid_rmse_mean, 'o-', color="green", label="Validation RMSE")

    # 绘制误差范围
    plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std, train_rmse_mean + train_rmse_std, alpha=0.2,
                     color="blue")
    plt.fill_between(train_sizes, valid_rmse_mean - valid_rmse_std, valid_rmse_mean + valid_rmse_std, alpha=0.2,
                     color="green")

    # 设置图形标题和标签
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title("Learning Curve for SVM Regressor (RMSE)")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig('./svm_trained.png')
    plt.show()

    # save model
    joblib.dump(svm_model, 'svm_trained.joblib')

    loaded_model = joblib.load('svm_trained.joblib')

    # 使用加载的模型进行预测
    y_loaded_pred = loaded_model.predict(X_test)

    return svm_model


@hydra.main(config_path="./", config_name="benchmark", version_base='1.3')
def benchmark(cfg: DictConfig) -> None:
    """
    """
    print('start to work ...')

    # ============================= models ======================

    # -------------- linear regressor --------------

    # -------------- ann --------------
    # ann_trained = ann_model_train(X_train_scaled, X_valid_scaled, X_test_scaled, cfg.model.ann_model_select)
    ann_trained = load_model(cfg.model.ann_model_select)

    # -------------- SVM --------------
    # svm_trained = svm_model_train(X_train_scaled, X_valid_scaled, X_test_scaled, cfg.model.svm_model_select)
    svm_trained= joblib.load(cfg.model.svm_model_select)

    # -------------- RF --------------
    # rf_trained = rf_model_train(cfg.model.rf_model_select)
    rf_trained, rf_version = joblib.load(cfg.model.rf_model_select)

    # -------------- xgb --------------
    xgb_default, xgboost_version, sklearn_version = joblib.load('../XGBoost/xgb_default.joblib')
    xgb_tuned, xgboost_version, sklearn_version = joblib.load(f'../XGBoost/xgb_tuned.joblib')

    # plot_learning_curve(xgb_default, evalSet=evalSet, fig_name='xgb_default.png')
    # plot_learning_curve(xgb_tuned, evalSet=evalSet, fig_name='xgb_tuned.png')

    # ============================= models ======================
    # loop foa all models:
    y_test2 = y_test.copy()
    model_list = [xgb_default, xgb_tuned, rf_trained, ann_trained, svm_trained]
    model_name = ['xgb_default', 'xgb_tuned', 'RF', 'ANN','SVM']
    for model, name in zip(model_list, model_name):

        print(f'------------------ {name} ----------')

        # make prediction:
        if name in ['ANN', 'SVM']:
            pred = model.predict(X_test_scaled)
        else:
            pred = model.predict(X_test)
        y_test2['CF_pred'] = pred

        # statistics with true values:
        stats: dict = calculate_statistics(y_test2)

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job))):
        print('start to work...')


if __name__ == "__main__":
    sys.exit(benchmark())
