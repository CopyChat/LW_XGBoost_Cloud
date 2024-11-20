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
from xgboost import XGBRegressor

import GEO_PLOT
import RESEARCH
import pickle
import PUBLISH
from sklearn.model_selection import train_test_split

# data:

X_train, y_train, X_valid, y_valid, X_test, y_test, \
    X_train_scaled, X_valid_scaled, X_test_scaled, evalSet = RESEARCH.prepare_data(train_valid_rate=0.2)


# ============================= done of read data ===========================
# functions:

def plot_learning_curve(model, scr_train, scr_test, feature_names, fig_name='learning_curve.png'):
    """
    using train and test valid datasets, but the showing "testing curve"
    model:
    evalSet:
    :return:
    """

    # plotting:

    result = model.evals_result()

    plt.figure(figsize=(8, 8))

    plt.plot(result['validation_0']['rmse'], label='train=%f' % scr_train, color='blue')
    plt.plot(result['validation_1']['rmse'], label='test=%f' % scr_test, color='red')

    # Add text to the top-left of the figure
    plt.text(0.15, 0.75, f'{len(feature_names)} features: {feature_names}',
             fontsize=12, color='green', transform=plt.gca().transAxes, verticalalignment='top')

    plt.tick_params(labelsize=14)
    plt.ylabel('', fontsize=148)
    plt.xlabel('', fontsize=148)
    plt.title("Learning curve RMSE (Stepwise_Feature_Elimination)", fontsize=14)
    plt.legend(prop={'size': 15})
    plt.ylim([0.06, 0.22])
    plt.savefig(f'./{fig_name:s}')
    plt.show()




@hydra.main(config_path="./", config_name="feature", version_base='1.3')
def feature(cfg: DictConfig) -> None:
    """
    """
    print('start to work ...')

    if cfg.job.Stepwise_Feature_Elimination:

        model = XGBRegressor(
            learning_rate=0.5, n_estimators=200,
            max_depth=5, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0, reg_lambda=0,
            base_score=0.5, booster='gbtree', importance_type='gain',
            interaction_constraints='', validate_parameters=1, verbosity=None)


        # change the order according to the feature importance from default xgboost model, figure in ./XGBoost
        features_to_keep =['LWdn', 'SWDif', 'SWDir', 'T', 'RH', 'GSW', 'P']

        # initial variables
        mab, rmse, cor, r_squared = [], [], [], []
        validation_accuracies = []
        train_accuracies = []

        # Iterate by reducing features
        for i in range(len(features_to_keep), 0, -1):
            # Select features to keep
            selected_features = features_to_keep[:i]
            X_train_reduced = X_train[selected_features]
            X_valid_reduced = X_valid[selected_features]
            X_test_reduced = X_test[selected_features]

            evalSet = [(X_train_reduced, y_train), (X_valid_reduced, y_valid)]

            print(f'start training...')
            model.fit(X_train_reduced, y_train, verbose=False, eval_set=evalSet)

            scr_train = round(model.score(X_train_reduced, y_train), 4)
            scr_valid = round(model.score(X_valid_reduced, y_valid), 4)

            plot_learning_curve(model, scr_train, scr_valid, selected_features, fig_name=f'{len(selected_features)}_features.png')

            #
            y_test['CF_pred'] = model.predict(X_test_reduced)

            stats: dict = RESEARCH.calculate_statistics_y_pred_y_test(y_test)

            train_accuracies.append(scr_train)
            validation_accuracies.append(scr_valid)
            mab.append(stats['MAB']),
            rmse.append(stats['RMSE'])
            cor.append(stats['COR'])
            r_squared.append(stats['R_squared'])

        plt.figure(figsize=(10, 6))
        statistics = [mab, rmse, cor, r_squared, train_accuracies, validation_accuracies]
        statistics_labels = ['MAB', 'RMSE', 'COR', 'R_squared (test data)', 'train_score_R^2', 'validation_score_R^2']
        x = range(len(features_to_keep), 0, -1)
        for i in range(len(statistics)):
            plt.plot(x, statistics[i], label=statistics_labels[i], marker='o')

            plt.xlabel("Number of Features")
            plt.ylabel("Accuracy")
            plt.title("Stepwise_Feature_Elimination: Model Performance")
            plt.legend()
        plt.grid()
        plt.ylim([0, 1])
        plt.text(0.15, 0.25, f'{len(features_to_keep)} features in order: {features_to_keep}',
                 fontsize=12, color='green', transform=plt.gca().transAxes, verticalalignment='top')
        # Add a text box
        model_params = f'default XGBoost model: \n'\
                       f'learning_rate=0.5, n_estimators=200,\n'\
                       f'max_depth=5, min_child_weight=3,\n'\
                       f'subsample=0.8, colsample_bytree=0.8,'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.45, 0.5, model_params, fontsize=12, color='black', transform=plt.gca().transAxes,
                 bbox=props, horizontalalignment='center', verticalalignment='center', wrap=True)

        plt.savefig(f'Stepwise_Feature_Elimination_xgb_default.png', dpi=220)
        plt.show()

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job))):
        print('start to work...')


if __name__ == "__main__":
    sys.exit(feature())
