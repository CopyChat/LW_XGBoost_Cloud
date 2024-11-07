"""
to apply ML method to improve cloud fraction estimation,
compared with traditional physical cloud method based on thermal physics,
where assumptions made in lower level of atmosphere.

This file is to analysis the XGBoost results, all ML works are deal with MS Code.
"""

__version__ = f'Version 1.0  \nTime-stamp: <2024-11-06>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
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
data_set_name = '../dataset/raw.bsrn_lacy.2019_2022.1min.local_time.csv'
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
evalSet = [(X_train, y_train), (X_valid, y_valid)]

# ============================= done of read data ===========================
# functions:

def my_learning_curve(model, X_train, X_test, y_train, y_test,fig_name):
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

def upgrade_model_and_plot(model, param_name, param_value):
    # upgrade model parameter:
    upgrade_model(model, param_name, param_value)

    model.fit(X_train, y_train, eval_set=evalSet)
    my_learning_curve(model, X_train, X_valid, y_train, y_valid, fig_name)

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


from scipy.stats import pearsonr


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

@hydra.main(config_path="./", config_name="benchmark", version_base='1.3')
def benchmark(cfg: DictConfig) -> None:
    """
    """
    print('start to work ...')

    # ============================= models ======================
    import xgboost as xgb

    xgb_default = pickle.load(open(f'../XGBoost/xgb_default.mat', 'rb'))
    xgb_tuned = pickle.load(open(f'../XGBoost/xgb_tuned.mat', 'rb'))

    for model in [xgb_default, xgb_tuned]:
        # learning curve:
        plot_learning_curve(xgb_default, evalSet=evalSet, fig_name='xgb_default.png')
        plot_learning_curve(xgb_tuned, evalSet=evalSet, fig_name='xgb_tuned.png')

        # make prediction:
        xgb_pred = xgb_default.predict(X_test)
        y_test['CF_pred'] = xgb_pred

        # statistics:
        stats: dict = calculate_statistics(y_test)

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job))):
        print('start to work...')


if __name__ == "__main__":
    sys.exit(benchmark())
