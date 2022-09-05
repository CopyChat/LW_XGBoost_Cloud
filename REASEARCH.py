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


