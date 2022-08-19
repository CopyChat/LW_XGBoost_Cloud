"""
data processing file
"""

import os
import scipy
import sys
from pathlib import Path
from typing import List
import warnings
import hydra
import seaborn as sns
from omegaconf import DictConfig
import cftime
import glob
import pandas as pd
import calendar
import numpy as np
from dateutil import tz
import xarray as xr
import cartopy.crs as ccrs

# to have the right backend for the font.
import matplotlib
import matplotlib.pyplot as plt

import GEO_PLOT
import cartopy.feature as cfeature
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.multitest import fdrcorrection as fdr_cor

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"


def figure_21_diurnal_curve_in_classif_MJO(classif: pd.DataFrame, field_1D: xr.DataArray,
                                           suptitle_add_word: str = '', anomaly: int = 0,
                                           percent: int = 0, ylimits='default', plot_big_data_test: int = 1):
    """

    Args:
        ylimits ():
        classif ():
        field_1D ():
        suptitle_add_word ():
        anomaly ():
        percent ():
        plot_big_data_test ():

    Returns:

    Applied_project:
     Mialhe_2020
    """
    # ----------------------------- data -----------------------------
    data_in_class = GEO_PLOT.get_data_in_classif(da=field_1D, df=classif, time_mean=False, significant=0)

    # to convert da to df: for the boxplot:

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))

    # ----------------------------- plot -----------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

    for i in range(len(class_names)):
        date_in_class = classif[classif['MJO_phase'] == class_names[i]].index.date
        data_1class = field_1D.loc[field_1D.time.dt.date.isin(date_in_class)]

        y = data_1class.groupby(data_1class['time'].dt.hour).mean()
        x = y.hour

        plt.plot(x, y, label=f'Phase_{i + 1:g}')

    plt.legend(bbox_to_anchor=(0.21, 0.43))
    plt.grid(True)

    plt.xlabel('Hour')

    if percent:
        plt.ylabel(f'percentage (%)')
    else:
        plt.ylabel(f'{data_in_class.name:s} ({data_in_class.units})')

    if ylimits != 'default':
        ax.set_ylim(ylimits[0], ylimits[1])

    # title = f'{field_1D.assign_attrs().long_name:s} percent={percent:g} anomaly={anomaly:g} in class'

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/Fig.21.diurnal_mjo_NDJF_high_lines.png', dpi=300)
    plt.show()
    print(f'got plot ')


def figure_19_diurnal_curve_in_classif(classif: pd.DataFrame, field_1D: xr.DataArray,
                                       suptitle_add_word: str = '',
                                       anomaly: int = 0,
                                       percent: int = 0,
                                       ylimits='default',
                                       plot_big_data_test: int = 1):
    """

    Args:
        ylimits ():
        classif ():
        field_1D ():
        suptitle_add_word ():
        anomaly ():
        percent ():
        plot_big_data_test ():

    Returns:

    Applied_project:
     Mialhe_2020
    """
    # ----------------------------- data -----------------------------
    data_in_class = GEO_PLOT.get_data_in_classif(da=field_1D, df=classif, time_mean=False, significant=0)

    # to convert da to df: for the boxplot:

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))

    # ----------------------------- plot -----------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

    for i in range(len(class_names)):
        date_in_class = classif[classif['class'] == class_names[i]].index.date
        data_1class = field_1D.loc[field_1D.time.dt.date.isin(date_in_class)]

        y = data_1class.groupby(data_1class['time'].dt.hour).mean()
        x = y.hour

        plt.plot(x, y, label=f'Reg_{i + 1:g}')

    plt.legend(bbox_to_anchor=(0.21, 0.43))
    plt.grid(True)

    plt.xlabel('Hour')

    if percent:
        plt.ylabel(f'percentage (%)')
    else:
        plt.ylabel(f'{data_in_class.name:s} ({data_in_class.units})')

    if ylimits != 'default':
        ax.set_ylim(ylimits[0], ylimits[1])

    # title = f'{field_1D.assign_attrs().long_name:s} percent={percent:g} anomaly={anomaly:g} in class'

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/Fig.19.diurnal_OLR_lines.png', dpi=300)
    plt.show()
    print(f'got plot ')


def figure_7_ssr_classification_MJO(field: xr.DataArray, classif: pd.DataFrame,
                                    vmax=400, vmin=-600, output='figure_7.png', bias=1,
                                    cbar_label=f'label', cmap=plt.cm.seismic,
                                    only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(12, 6), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.8, wspace=0.02, hspace=0.02)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            # tag = f'Cl{cls+1:g}'
            # ax.text(0.98, 0.98, f'{tag:s}', fontsize=12,
            #         horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            # mean_value = geomap.mean().values
            # if np.abs(mean_value) > 0:
            #     ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=12,
            #             horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # ===========================================

            if cls == 0:
                ax.set_title(f'{hours[hour]:g}:00', fontsize=18)

            if cls == 8:
                ax.set_xlabel(f'{hours[hour]:g}:00', fontsize=18)

            mean_value = np.float(data_1h_mean.mean())

            if cls + 1 == 2:
                ax.text(0.99, 0.001, f'{mean_value:4.2f}', fontsize=15, color='white',
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
            else:
                if np.abs(mean_value) > 0:
                    ax.text(0.99, 0.001, f'{mean_value:4.2f}', fontsize=15,
                            horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # to set y label
    for i in range(n_class):
        ax = axs[i, 0]
        plt.sca(ax)
        plt.ylabel(f'CL{str(class_names[cls]):s}')
        ax.set_ylabel(f'CL{str(class_names[cls]):s}')

    # the above code doesn't work, so:

    y_label = ''
    for i in range(n_class):
        y_label += f'CL{class_names[i]:g}        '
        # y_label += f'CL{class_names[i]:g} ({class_size[class_names[i]]/10:g}) '
    y_label = f'Pha_7    ' \
              f'Pha_6   ' \
              f'Pha_5    ' \
              f'Pha_4    '

    # without num:
    plt.figtext(0.075, 0.23, y_label, rotation='vertical', fontsize=16)
    plt.figtext(0.045, 0.35, 'MJO phases NDJF', rotation='vertical', fontsize=16)
    # with num:
    # plt.figtext(0.08, 0.11, y_label, rotation='vertical', fontsize=20)
    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.15, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)
    print(f'got plot ')


def figure_1_ssr_classification_OLR(field: xr.DataArray, classif: pd.DataFrame,
                                    vmax=400, vmin=-600, output='figure_1.png', bias=1,
                                    cbar_label=f'label', cmap=plt.cm.seismic,
                                    only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(14, 9), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.02, hspace=0.02)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            # tag = f'Cl{cls+1:g}'
            # ax.text(0.98, 0.98, f'{tag:s}', fontsize=12,
            #         horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            # mean_value = geomap.mean().values
            # if np.abs(mean_value) > 0:
            #     ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=12,
            #             horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # ===========================================

            if cls == 0:
                ax.set_title(f'{hours[hour]:g}:00', fontsize=18)

            if cls == 8:
                ax.set_xlabel(f'{hours[hour]:g}:00', fontsize=18)

            mean_value = np.float(data_1h_mean.mean())

            if cls + 1 == 2:
                ax.text(0.99, 0.001, f'{mean_value:4.2f}', fontsize=15, color='white',
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
            else:
                if np.abs(mean_value) > 0:
                    ax.text(0.99, 0.001, f'{mean_value:4.2f}', fontsize=15,
                            horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # to set y label
    for i in range(n_class):
        ax = axs[i, 0]
        plt.sca(ax)
        plt.ylabel(f'CL{str(class_names[cls]):s}')
        ax.set_ylabel(f'CL{str(class_names[cls]):s}')

    # the above code doesn't work, so:

    y_label = ''
    for i in range(n_class):
        y_label += f'CL{class_names[i]:g}        '
        # y_label += f'CL{class_names[i]:g} ({class_size[class_names[i]]/10:g}) '
    y_label = f'Reg_7     ' \
              f'Reg_6    ' \
              f'Reg_5     ' \
              f'Reg_4    ' \
              f'Reg_3    ' \
              f'Reg_2    ' \
              f'Reg_1     '

    # without num:
    plt.figtext(0.075, 0.125, y_label, rotation='vertical', fontsize=18)
    # with num:
    # plt.figtext(0.08, 0.11, y_label, rotation='vertical', fontsize=20)
    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)
    print(f'got plot ')


def ssr_cluster_in_month_hour2(field, classif, vmax, vmin, cbar_label, output, only_sig, cmap, bias):
    field_classif = GEO_PLOT.get_data_in_classif(da=field, df=classif)
    # ----------------------------- definition -----------------------------
    fontsize = 14
    # ----------------------------- definition -----------------------------

    cluster_names = sorted(list(set(classif.values.ravel())))
    n_cluster = len(cluster_names)

    # ----------------------------- plot -----------------------------
    import matplotlib.gridspec as gridspec
    from cartopy.mpl.geoaxes import GeoAxes
    import cartopy.crs as ccrs
    from mpl_toolkits.axes_grid1 import AxesGrid

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))

    fig = plt.figure(figsize=(26, 30), dpi=300)

    for c in range(n_cluster):

        field_in_class = field_classif.where(field_classif['class'] == cluster_names[c],
                                             drop=True).squeeze().dropna('time')

        # ---------- data ----------
        months = sorted(list(set(field_in_class.time.dt.month.values)))
        hours = sorted(list(set(field_in_class.time.dt.hour.values)))

        # ------------- plot inner grid -------------
        axgr = AxesGrid(fig, (3, 3, c + 1), axes_class=axes_class,
                        nrows_ncols=(12, 10),
                        axes_pad=0.6,
                        cbar_location='right',
                        cbar_mode='single',
                        cbar_pad=0.2,
                        cbar_size='3%',
                        label_mode='')  # note the empty label_mode

        n_day_in_month = np.zeros((12,)).astype(np.int)
        for mm in range(len(months)):

            print(f'plot month = {months[mm]:g}')
            in_month = field_in_class.where(field_in_class.time.dt.month == months[mm], drop=True).squeeze()
            n_day = int(len(in_month) / len(hours))
            n_day_in_month[mm] = n_day

            for hh in range(len(hours)):

                print(f'plot in hour = {hh + 8:g}')

                # get ax:
                ax_index = mm * len(hours) + hh

                ax = axgr[ax_index]

                print(mm, hh, ax_index)

                data_in_hour = in_month.where(in_month.time.dt.hour == hours[hh], drop=True)
                hourly_mean = data_in_hour.mean('time')

                if only_sig:
                    sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                            fdr_correction=1)
                    data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
                else:
                    data_1h = data_in_hour

                data_1h_mean = data_1h.mean('time')
                # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
                # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
                # got an error.

                # start to plot in each hour:
                # ===========================================
                # set up map:
                GEO_PLOT.set_basemap(ax, area='reu')

                cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

                geomap = data_1h_mean

                cf = ax.pcolormesh(geomap.lon, geomap.lat, geomap,
                                   cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

                ax.text(0.98, 0.98, f'test', fontsize=fontsize,
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        # # ----------------------------- end of plot -----------------------------

        cb_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
        cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
        cb.ax.tick_params(labelsize=16)
        cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)
    plt.show()
    print(f'got plot ')


def ssr_cluster_in_month_hour(field, classif, vmax, vmin, cbar_label, output, only_sig, cmap, bias):
    field_classif, size = GEO_PLOT.get_data_in_classif(da=field, df=classif, return_size=True)
    # ----------------------------- definition -----------------------------
    fontsize = 15
    # ----------------------------- definition -----------------------------

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cluster_names = sorted(list(set(classif.values.ravel())))
    n_cluster = len(cluster_names)

    # ----------------------------- plot -----------------------------
    fig = plt.figure(figsize=(26, 28), dpi=300)

    import matplotlib.gridspec as gridspec
    outer_grid = gridspec.GridSpec(3, 3, wspace=0.25, hspace=0.25)  # 9 clusters

    # --------------------------- cluster title in bold:
    cluster_size = (np.array(list(size.values())) / 10).astype(np.int)
    labels = np.reshape([f'CL{x + 1:g} ({cluster_size[x]:g})' for x in range(n_cluster)], (3, 3), order='C')

    y_offset_fraction = 0.02
    for i in range(3):
        for j in range(3):
            y_position = outer_grid[i, j].get_position(fig).p1[1] + y_offset_fraction
            x_position = outer_grid[i, j].get_position(fig).p0[0] + outer_grid[i, j].get_position(fig).bounds[2] / 2
            # p0 is the left bottom point of box, p1 is the right top point
            print(labels[i, j], x_position, y_position)
            fig.text(x_position, y_position, labels[i, j], ha='center', va='center',
                     style='normal', fontweight='bold', fontsize=fontsize)

    for c in range(n_cluster):

        field_in_class = field_classif.where(field_classif['class'] == cluster_names[c],
                                             drop=True).squeeze().dropna('time')

        # ---------- data ----------
        months = sorted(list(set(field_in_class.time.dt.month.values)))
        hours = sorted(list(set(field_in_class.time.dt.hour.values)))

        # ------------- plot inner grid -------------
        inner_grid = gridspec.GridSpecFromSubplotSpec(len(months), len(hours), outer_grid[c],
                                                      hspace=0.01, wspace=0.01)

        n_day_in_month = np.zeros((12,)).astype(np.int)
        for mm in range(len(months)):
            print(f'plot month = {months[mm]:g}')
            in_month = field_in_class.where(field_in_class.time.dt.month == months[mm], drop=True).squeeze()
            n_day = int(len(in_month) / len(hours))
            n_day_in_month[mm] = n_day

            for hh in range(len(hours)):

                ax = plt.subplot(inner_grid[mm, hh], projection=ccrs.PlateCarree())

                data_in_hour = in_month.where(in_month.time.dt.hour == hours[hh], drop=True)

                if only_sig:
                    sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour, fdr_correction=1)
                    data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
                else:
                    data_1h = data_in_hour

                data_1h_mean = data_1h.mean('time')
                # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
                # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
                # got an error.

                # start to plot in each hour:
                # ===========================================
                # set up map:
                GEO_PLOT.set_basemap(ax, area='reu')

                cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

                geomap = data_1h_mean

                cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                    cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

                font_kwargs = dict(fontfamily="sans-serif", fontweight="regular", fontsize=fontsize)

                if hh == 0:
                    ax.annotate(
                        f'{month_names[mm]:s} ({n_day:g})',
                        xy=(0, 0.5), xycoords=ax.yaxis.label,
                        xytext=(-0.2, 0.5), textcoords="axes fraction",
                        ha="right", va="center", rotation=0, **font_kwargs)

                # hours on the top
                if mm == 0:
                    ax.set_title(
                        f'{hours[hh]:g}H', fontsize=fontsize)

                # hours on the bottom
                if mm + 1 == len(months):
                    ax.annotate(
                        f'{hours[hh]:g}H',
                        xy=(0.5, 1), xycoords="axes fraction",
                        xytext=(0.5, -0.2), textcoords="axes fraction",
                        ha="center", va="top", rotation=0, **font_kwargs)

                # text:
                # ax.text(0.98, 0.98, f'test', fontsize=fontsize,
                #         horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.06, 0.7, 0.015])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)
    # plt.show()
    print(f'got plot ')


def persistence(classif: pd.DataFrame, plot: bool = True):
    from datetime import timedelta

    column_name = classif.columns.values[0]
    class_names = np.sort(list(set(classif.values.ravel())))
    n_class = len(class_names)
    n_day = len(classif)

    persistence_all_class = []
    max_d = 1
    for i in range(n_class):
        # select 1st
        persistence_class = []
        class1 = pd.DataFrame(classif[classif[column_name] == class_names[i]])
        class1_index = class1.index

        j = 0
        while j < len(class1):
            print(f'starting index: {j:g}')
            today = class1.index[j]
            next_days = today + timedelta(seconds=3600 * 24)

            d = 1
            while class1_index.isin([next_days]).any():
                d += 1
                next_days += timedelta(seconds=3600 * 24)
                if d > max_d:
                    max_d = d
            else:
                j += d

            print(f'start of event=', today, f'next start={j:g}',
                  f'next start:', next_days, f'duration = ', d)
            persistence_class.append(d, )
        persistence_all_class.append(persistence_class)

    # this calculation could be checked by:
    # sum(persistence_in_class) = n_day in the class

    # prepare a 2D output:
    table = np.zeros((n_class, max_d))
    for i in range(n_class):
        # select 1st
        persistence_class = persistence_all_class[i]
        n_event = len(persistence_class)

        count = pd.value_counts(persistence_class).sort_index() * 100 / n_event
        print(i + 1, count.index, count.values)

        # put values in the table:

        x = list(count.index.values)

        for jj in range(len(x)):
            position_x = x[jj] - 1
            value_x = count.values[jj]
            table[i, position_x] = value_x

        # create pd.DataFrame
        df = pd.DataFrame(data=table, index=[f'CL{i + 1:g}' for i in range(n_class)],
                          columns=[f'{i + 1:g}' for i in range(max_d)])

    if plot:
        styles = ['solid', 'dashed',
                  'dotted', 'dotted',
                  'solid', 'dashed', 'dashed',
                  'solid', 'solid']
        colors = ['black', 'blue',
                  'red', 'red',
                  'black', 'blue', 'blue',
                  'black', 'black']

        # a normal plot:
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        for i in range(n_class):
            # select 1st
            persistence_class = persistence_all_class[i]
            n_event = len(persistence_class)

            count = pd.value_counts(persistence_class).sort_index() * 100 / n_event
            print(i + 1, count.index, count.values)

            plt.plot(count.index, count.values, label=f'CL{i + 1:g}', marker='o',
                     color=colors[i], linestyle=styles[i])

        ax.set_yscale('log')
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.grid()
        plt.legend()
        plt.xlabel(f'duration (day)')
        plt.ylim(0.05, 85)
        plt.ylabel(f'percentage of all periods Class event (%)')
        plt.savefig(f'./plot/persistence_SSR_class.png', dpi=300)
        plt.show()

        # a matrix plot
        # plot only duration less than 5 days
        df_plot = df[['1', '2', '3', '4', '5']]
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        GEO_PLOT.plot_color_matrix(df_plot, cbar_label='%', ax=ax, plot_number=True, cmap='Greens')
        plt.savefig(f'./plot/Figure.persistence.png', dpi=300)
        plt.show()

    return df


def figure_3_ssr_classification_clearsky(field: xr.DataArray, classif: pd.DataFrame,
                                         vmax=400, vmin=-600, output='figure_1.png', bias=1,
                                         cbar_label=f'label', cmap=plt.cm.seismic,
                                         add_triangle: bool = True,
                                         only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    land_mask = np.load('./dataset/sarah_e.land_mask.format_pauline.mat.npy')
    lookup = xr.DataArray(land_mask, dims=('y', 'x'))
    # ----------------------------- get definitions -----------------------------
    var_name = field.name
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(20, 16), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.02, hspace=0.02)

    # add headers:
    row_headers = [f'CL{x + 1:g}' for x in range(n_class)]
    col_headers = [f'{x + 4:g}:00' for x in range(21)]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=20)
    font_kwargs = dict(fontfamily="sans-serif", fontweight="regular", fontsize=20)
    GEO_PLOT.fig_add_headers(fig, row_headers=row_headers, col_headers=col_headers,
                             rotate_row_headers=True, **font_kwargs)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            if add_triangle:
                # ---------------- add triangles for percentile ---------
                # find land only field:

                geo_land = geomap.where(lookup)
                # test
                geo_df = geo_land.to_dataframe().dropna()

                plot_percentile = False

                if plot_percentile:
                    # if plot 10 percentile, may have different num of markers, even zero, if not normal distribution
                    limits = GEO_PLOT.get_confidence_interval(geo_df[var_name], 0.1)

                    for i in range(len(geo_df)):
                        if geo_df['SIS'].values[i] < limits[0]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='v', edgecolor='purple', color='white')
                        if geo_df['SIS'].values[i] > limits[1]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='^', edgecolor='black', color='red')

                else:
                    # if plot 10 % of max and min values
                    geo_df_sort = geo_df.sort_values(var_name)
                    lon = geo_df_sort['lon'].values
                    lat = geo_df_sort['lat'].values

                    limit = round(len(geo_df_sort) * 0.1)

                    for i in range(limit):
                        plt.scatter(lon[i], lat[i], s=35, marker='v', edgecolor='purple', color='white')
                        plt.scatter(lon[len(geo_df_sort) - 1 - i],
                                    lat[len(geo_df_sort) - 1 - i],
                                    s=35, marker='^', edgecolor='black', color='red')

                # ---------------- end

            # ===========================================
            mean_value = np.float(data_1h_mean.mean())

            ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16,
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)

    if add_triangle:
        geo_land.plot()
        plt.show()
        print(f'got plot ')


def figure_3_ssr_classification_clearsky_era5_data_only(field: xr.DataArray, classif: pd.DataFrame,
                                                        vmax=400, vmin=-600, output='figure_1.png', bias=1,
                                                        cbar_label=f'label', cmap=plt.cm.seismic,
                                                        add_triangle: bool = True,
                                                        only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    land_mask = np.load('./dataset/sarah_e.land_mask.format_pauline.mat.npy')
    lookup = xr.DataArray(land_mask, dims=('y', 'x'))
    # ----------------------------- get definitions -----------------------------
    var_name = field.name
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(20, 20), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.02, hspace=0.02)

    # add headers:
    row_headers = [f'CL{x + 1:g}' for x in range(n_class)]
    col_headers = [f'{x + 4:g}:00' for x in range(21)]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=20)
    font_kwargs = dict(fontfamily="sans-serif", fontweight="regular", fontsize=20)
    GEO_PLOT.fig_add_headers(fig, row_headers=row_headers, col_headers=col_headers,
                             rotate_row_headers=True, **font_kwargs)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            add_triangle = False

            if add_triangle:
                # ---------------- add triangles for percentile ---------
                # find land only field:

                geo_land = geomap.where(lookup)
                # test
                geo_df = geo_land.to_dataframe().dropna()

                plot_percentile = False

                if plot_percentile:
                    # if plot 10 percentile, may have different num of markers, even zero, if not normal distribution
                    limits = GEO_PLOT.get_confidence_interval(geo_df[var_name], 0.1)

                    for i in range(len(geo_df)):
                        if geo_df['SIS'].values[i] < limits[0]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='v', edgecolor='purple', color='white')
                        if geo_df['SIS'].values[i] > limits[1]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='^', edgecolor='black', color='red')

                else:
                    # if plot 10 % of max and min values
                    geo_df_sort = geo_df.sort_values(var_name)
                    lon = geo_df_sort['lon'].values
                    lat = geo_df_sort['lat'].values

                    limit = round(len(geo_df_sort) * 0.1)

                    for i in range(limit):
                        plt.scatter(lon[i], lat[i], s=35, marker='v', edgecolor='purple', color='white')
                        plt.scatter(lon[len(geo_df_sort) - 1 - i],
                                    lat[len(geo_df_sort) - 1 - i],
                                    s=35, marker='^', edgecolor='black', color='red')

                # ---------------- end

            # ===========================================
            mean_value = np.float(data_1h_mean.mean())

            ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16,
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)

    if add_triangle:
        geo_land.plot()
        plt.show()
        print(f'got plot ')


def figure_1_ssr_classification(field: xr.DataArray, classif: pd.DataFrame,
                                vmax=400, vmin=-600, output='figure_1.png', bias=1,
                                cbar_label=f'label', cmap=plt.cm.seismic,
                                add_triangle: bool = True,
                                only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    land_mask = np.load('./dataset/sarah_e.land_mask.format_pauline.mat.npy')
    lookup = xr.DataArray(land_mask, dims=('y', 'x'))
    # ----------------------------- get definitions -----------------------------
    var_name = field.name
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(20, 16), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.02, hspace=0.02)

    # add headers:
    row_headers = [f'CL{x + 1:g}' for x in range(n_class)]
    col_headers = [f'{x + 8:g}:00' for x in range(10)]
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=20)
    font_kwargs = dict(fontfamily="sans-serif", fontweight="regular", fontsize=20)
    GEO_PLOT.fig_add_headers(fig, row_headers=row_headers, col_headers=col_headers,
                             rotate_row_headers=True, **font_kwargs)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            if add_triangle:
                # ---------------- add triangles for percentile ---------
                # find land only field:

                geo_land = geomap.where(lookup)
                # test
                geo_df = geo_land.to_dataframe().dropna()

                plot_percentile = False

                if plot_percentile:
                    # if plot 10 percentile, may have different num of markers, even zero, if not normal distribution
                    limits = GEO_PLOT.get_confidence_interval(geo_df[var_name], 0.1)

                    for i in range(len(geo_df)):
                        if geo_df['SIS'].values[i] < limits[0]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='v', edgecolor='purple', color='white')
                        if geo_df['SIS'].values[i] > limits[1]:
                            plt.scatter(geo_df['lon'].values[i], geo_df['lat'].values[i], s=20,
                                        marker='^', edgecolor='black', color='red')

                else:
                    # if plot 10 % of max and min values
                    geo_df_sort = geo_df.sort_values(var_name)
                    lon = geo_df_sort['lon'].values
                    lat = geo_df_sort['lat'].values

                    limit = round(len(geo_df_sort) * 0.1)

                    for i in range(limit):
                        plt.scatter(lon[i], lat[i], s=35, marker='v', edgecolor='purple', color='white')
                        plt.scatter(lon[len(geo_df_sort) - 1 - i],
                                    lat[len(geo_df_sort) - 1 - i],
                                    s=35, marker='^', edgecolor='black', color='red')

                # ---------------- end

            # ===========================================
            mean_value = np.float(data_1h_mean.mean())

            if cls + 1 == 7:
                ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16, color='white',
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
            else:
                ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16,
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)

    if add_triangle:
        geo_land.plot()
        plt.show()
        print(f'got plot ')


def figure2_monthly_num_ssr_class(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)

    styles = ['solid', 'dashed',
              'dotted', 'dotted',
              'solid', 'dashed', 'dashed',
              'solid', 'solid']
    colors = ['black', 'blue',
              'red', 'red',
              'black', 'blue', 'blue',
              'black', 'black']

    monthly_all = df.groupby(df.index.month).count().values.ravel()
    for c in range(9):
        class_name = c + 1
        df_c = df[df['9Cl'] == class_name]

        c_count = df_c.groupby(df_c.index.month).count().values.ravel()
        ys = [c_count[i] * 100 / monthly_all[i] for i in range(12)]

        xs = np.arange(1, 13)
        plt.plot(xs, ys,
                 color=colors[c],
                 linestyle=styles[c], linewidth=2)

        for x, y in zip(xs, ys):
            plt.text(x, y, str(class_name), color=colors[c], fontsize=12)

        # plot a cycle
        y_max = max(ys)
        x_max = xs[ys.index(y_max)]

        circle = plt.Circle((x_max, y_max), 24, color=colors[c], fill=True)

    ax.set_xticks(xs)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax.set_xticklabels(months)

    # plt.xlabel('Month')
    plt.ylabel('monthly occurrence (%)')

    plt.savefig(f'./plot/figure2.monthly.ssr_cluster.png', dpi=300)
    plt.show()
    print(f'got plot')


def figure_1_ssr_classification_for_cyclone(field: xr.DataArray, classif: pd.DataFrame,
                                            vmax=400, vmin=-600, output='figure_1.png', bias=1,
                                            cbar_label=f'label', cmap=plt.cm.seismic,
                                            only_sig: bool = False):
    # ----------------------------- data -----------------------------
    data_in_class, class_size = GEO_PLOT.get_data_in_classif(
        da=field, df=classif, time_mean=False, significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(16, 4), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.95, wspace=0.01, hspace=0.01)

    # ------- test the label first ----------
    row_headers = [f'no cyclone\n(2140)', f'cyclone\n(68)']
    col_headers = [f'{x + 8:g}:00' for x in range(10)]

    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=18)
    GEO_PLOT.fig_add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

    # ------- test the label first ----------

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_sig:
                sig_map = GEO_PLOT.value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                                        fdr_correction=1)
                data_1h = GEO_PLOT.filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            # start to plot in each hour:
            # ===========================================
            plt.sca(ax)
            # active this subplot

            # set up map:
            GEO_PLOT.set_basemap(ax, area='reu')

            cmap, norm = GEO_PLOT.set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

            geomap = data_1h_mean

            cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                                cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

            # ===========================================
            mean_value = np.float(data_1h_mean.mean())
            ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16,
                    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

            # num of record
            # ax.text(0.01, 0.01, f'{num_record/10:g}', fontsize=16,
            #         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    # ----------------------------- end of plot -----------------------------

    cb_ax = fig.add_axes([0.15, 0.15, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=16)
    cb.set_label(label=cbar_label, fontsize=16)

    plt.savefig(f'./plot/{output}', dpi=300)
    print(f'got plot ')
