"""
to analyis cloud type classification from SAFNWC
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import glob
import hydra
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from omegaconf import DictConfig
from importlib import reload
import pandas as pd

import GEO_PLOT

from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_mean_by_interval(df, interval='10min', min_point_inside_interval=5):
    """
    Resample a DataFrame to a specified time interval and calculate the mean value for each interval.
    :param df: DataFrame with a datetime index
    :param min_point_inside_interval: minimum number of points inside the interval
    :param interval: Time interval for resampling (default is 10 minutes)
    :return: DataFrame with mean values for each interval
    """
    # Resample to 10-minute intervals with right label (ending at 0, 10, 20, etc.)
    resampled_df = df.resample(interval, label='right', closed='right').mean()

    # Count number of available rows in each 10-minute interval
    counts = df.resample(interval, label='right', closed='right').count()

    # Identify intervals with less than 5 data points
    insufficient_data_intervals = counts[counts.iloc[:, 0] < min_point_inside_interval].index

    # Remove intervals with less than 5 data points
    cleaned_df = resampled_df.drop(insufficient_data_intervals)

    # Print missing times
    # print("Missing time intervals with insufficient data:")
    # for missing_time in insufficient_data_intervals:
        # print(missing_time)

    # Count total number of missing intervals
    missing_count = len(insufficient_data_intervals)
    print(f"\nTotal number of missing 10-minute intervals: {missing_count}")

    # Create a histogram of missing intervals
    missing_hours = [ts.hour for ts in insufficient_data_intervals]
    missing_minutes = [ts.minute for ts in insufficient_data_intervals]
    
    # Combine hours and 10-minute intervals for histogram bins
    missing_bins = [h * 6 + (m // 10) for h, m in zip(missing_hours, missing_minutes)]
    
    plt.figure(figsize=(12, 6))
    plt.hist(missing_bins, bins=range(0, 24 * 6 + 1), edgecolor='black', align='left')
    plt.xticks(ticks=range(0, 24 * 6, 6), labels=[f"{h}:00" for h in range(24)], rotation=45)
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Missing 10-Minute Intervals')
    plt.title('Histogram of Missing 10-Minute Intervals Throughout the Day')
    plt.tight_layout()
    plt.show()

    return cleaned_df


def split(data, tst_sz, shuffle=True):
    y = data["CF"]
    X = data.drop("CF" , axis=1)
    # X = X.drop("timestamp" , axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=tst_sz, random_state=7, shuffle=shuffle)
    return X_train, X_test, y_train, y_test

# Global data:
# Global constant definition (naming in uppercase)
def prepare_data(df_raw=None, train_valid_rate=0.1, with_time=False, shuffle=True):
    """
    get data ready for ML models
    :return:
    """
    if df_raw is None:
        # use default data 5 minute
        # ============================= read data ===========================
        path = f'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/LW_XGBoost_Cloud'
        data_set_name = f'{path}/dataset/raw.bsrn_lacy.2019_2022.5min.local_time.csv'
        df_raw = GEO_PLOT.read_csv_into_df_with_header(data_set_name)

    # ----------------------------  split data:
    if with_time:
        # Add 'dayofyear' and 'hourofday' periodic sine values
        df_raw['dayofyear'] = np.sin(2 * np.pi * df_raw.index.dayofyear / 365.0)
        df_raw['hourofday'] = np.sin(2 * np.pi * df_raw.index.hour / 24.0)

    # ----------------------------  split data:
    # works on two-year data:
    # for training (SearchGrid and CV) and valid
    train_valid = df_raw['2019-09-13':'2021-09-12']
    X_train, X_valid, y_train, y_valid = split(train_valid, train_valid_rate, shuffle=shuffle)

    # valid set do not directly participate in the training, but only for monitoring and validation and early_stop.

    predictors = list(df_raw.columns)
    # #predictors.remove('P')
    # #predictors.remove('RH')
    predictors.remove('CF')

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

    return X_train, y_train, X_valid, y_valid,  X_test, y_test, X_train_scaled, X_valid_scaled, X_test_scaled, evalSet, df_raw


def calculate_statistics_y_pred_y_test(df, print_out=False):
    """
    default columns names are 'CF', 'CF_pred'
    :param df:
    :return:
    """
    from scipy.stats import pearsonr

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

    if print_out:
    # Print the results with 2.2f format, aligned
        print("\nStatistics:")
        for key, value in output.items():
            print(f"{key:<20}: {value:6.2f}")

    return output


def compare_curves(df: pd.DataFrame, output_tag: str = None):
    var = df.columns.to_list()
    fig, ax = plt.subplots(1, figsize=(20, 5), dpi=300)
    x = range(len(df))
    x_ticks = x[::12 * 12]
    x_ticks_label = df.index.strftime("%Y-%m-%d %HH")[::12 * 12]
    for i in range(len(var)):
        plt.plot(x, df[var[i]], label=var[i])

    ax.set_xticks(x_ticks, minor=False)
    ax.set_xticklabels(x_ticks_label, minor=False)
    plt.xticks(rotation=15)
    plt.title(f'6AM to 6PM')
    plt.legend()
    plt.savefig(f'./plot/valid.1.{output_tag:s}.png', dpi=300)
    plt.show()


def read_mino_results(file_path: str):

    # read
    result_mino = GEO_PLOT.read_csv_into_df_with_header(file_path)
    # shift to local time
    result_mino = GEO_PLOT.convert_df_shifttime(result_mino, 3600 * 4)
    # select only 6AM to 6PM. (already done by Mino, just to confirm)
    result_mino = result_mino.between_time('6:00', '18:00')

    df_valid = result_mino[{'CF_XGB', 'CF_APCADA', 'CF_OBS', 'PCA_APCADA'}]

    return df_valid


def plot_corr(df: pd.DataFrame):
    cor = df.corr(method='pearson')

    # plot:
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot()
    GEO_PLOT.plot_color_matrix(df=cor, cmap=plt.cm.get_cmap('PiYG').reversed(), plot_number=True, ax=ax,
                               vmin=-1, vmax=1,
                               cbar_label='jj')
    plt.savefig(f'./plot/valid.2.cross_corr.png', dpi=300)
    plt.show()


def check_normal(df: pd.DataFrame, output_tag: str = 'output_tag'):

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(range(df.size), df.values)
    plt.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    df.hist(bins=10, alpha=0.5, ax=ax2)
    df.plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid(zorder=-1)

    plt.savefig(f'./plot/check_normal.{output_tag:s}.png', dpi=300)
    plt.show()

    from scipy import stats
    print(stats.normaltest(df.values))


def add_ct_for_validation(df_valid, ct, output: str = None):

    ct_nearest_list = []
    dt_nearest_list = []
    for i in range(len(df_valid)):
        index = ct.index.get_loc(df_valid.index[i], method='nearest')
        ct_nearest = ct.iloc[index]
        print(i, ct_nearest.values)
        ct_nearest_list.append(ct_nearest[0])
        dt_nearest_list.append(ct.index[index])

    df_valid.insert(0, 'ct', ct_nearest_list)
    df_valid.insert(0, 'dt_saf_nwc', dt_nearest_list)
    df_valid.insert(0, 'dt_delta', np.abs(df_valid.index - df_valid.dt_saf_nwc))
    df_valid.insert(0, 'diff_minute', [x.seconds % 3600 // 60 for x in df_valid.dt_delta])

    df_valid.to_pickle('./dataset/data_valid_ct')

    return df_valid


def valid_by_octas(df):
    df.insert(0, 'bias_XGB_octas', df['XGB_octas'] - df['OBS_octas'])
    df.insert(0, 'bias_APCADA_octas', df['PCA_APCADA'] - df['OBS_octas'])

    # plot:
    df[{'bias_APCADA_octas', 'bias_XGB_octas'}].apply(pd.Series.value_counts).plot(y=['bias_XGB_octas', 'bias_APCADA_octas'], kind='bar')
    plt.savefig(f'./plot/valid_by_octas.png', dpi=300)
    plt.show()

    # plot 2:

    total_n = len(df)

    bias_1 = []
    bias_2 = []
    for i in range(9):
        bias_n2 = len(df[np.abs(df['bias_APCADA_octas']) <= i]) * 100 / total_n
        bias_n1 = len(df[np.abs(df['bias_XGB_octas']) <= i]) * 100 / total_n

        print(f'bias <= {i:g} = {bias_n1: 4.2f}%, {bias_n2: 4.2f}%')

        bias_1.append(bias_n1)
        bias_2.append(bias_n2)

    x = np.array(range(9))
    plt.bar(x - 0.1, width=0.2,height=bias_1, color='blue', label='XGB_octas')
    plt.bar(x + 0.1, width=0.2,height=bias_2, color='orange', label='APCADA_octas')

    plt.ylim([0, 120])
    plt.ylabel('%')
    plt.xlabel('bias absolute in octas')
    plt.title(f'frequency of absolute bias')
    plt.legend(loc='upper left')

    plt.savefig(f'./plot/valid_by_octas_bias_frequency.png', dpi=300)
    plt.show()


