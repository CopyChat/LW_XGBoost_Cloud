"""
to apply ML method to improve cloud fraction estimation,
compared with traditional physical cloud method based on thermal physics,
where assumptions made in lower level of atmosphere.
"""

__version__ = f'Version 2.0  \nTime-stamp: <2022-08-19>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
import numpy as np
# import subprocess
# import numpy as np
# import pandas as pd
# import xarray as xr
# from importlib import reload
from importlib import reload
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd

import GEO_PLOT
import REASEARCH
import PUBLISH


@hydra.main(config_path="configs", config_name="config")
def cloud(cfg: DictConfig) -> None:
    """
    """
    print('start to work ...')
    # read
    df_raw = GEO_PLOT.read_csv_into_df_with_header(cfg.file.result_mino)
    # shift to local time
    df_raw = GEO_PLOT.convert_df_shifttime(df_raw, 3600 * 4)
    # select only 6AM to 6PM. (already done by Mino, just to confirm)
    df_raw = df_raw.between_time('6:00', '18:00')

    df_valid = df_raw[{'CF_XGB', 'CF_APACADA', 'CF_OBS'}]

    # plot first week:
    REASEARCH.compare_curves(df_valid['2021-01-04': '2021-01-10'], output_tag='raw')

    # correlation:
    REASEARCH.plot_corr(df_valid)
    cor = df_valid.corr()
    print(cor)

    # smooth of CF_APACADA
    from scipy.signal import savgol_filter

    df_valid.insert(0, 'CF_APACADA_smooth', savgol_filter(df_valid.CF_APACADA, window_length=25, polyorder=3))
    # ct_learn: replace value with condition:
    df_valid = df_valid.where(df_valid['CF_APACADA_smooth'] <= 1, other=1)

    REASEARCH.compare_curves(df_valid[{'CF_APACADA_smooth', 'CF_APACADA'}]['2021-01-04': '2021-01-07'],
                             output_tag='smoothed')
    cor = df_valid[{'CF_APACADA_smooth', 'CF_APACADA', 'CF_XGB', 'CF_OBS'}].corr()
    print(cor)

    # ct_learn: new column
    df_valid.insert(0, column='bias_XGB', value=df_valid['CF_XGB'] - df_valid['CF_OBS'])
    df_valid = df_valid.assign(bias_APACADA=df_valid['CF_APACADA'] - df_valid['CF_OBS'])

    REASEARCH.compare_curves(df_valid['2021-01-04': '2021-01-07'], output_tag='smoothed_with_bias')
    cor = df_valid[{'CF_APACADA_smooth', 'CF_APACADA', 'CF_OBS', 'CF_XGB'}].corr()
    print(cor)

    # normality
    REASEARCH.check_normal(df_valid[{'CF_APACADA'}], output_tag='check.normal')
    REASEARCH.check_normal(df_valid[{'CF_OBS'}], output_tag='CF_OBS')
    REASEARCH.check_normal(df_valid[{'CF_XGB'}], output_tag='check.normal')

    if cfg.job.valid.by_hour:
        # bias distribution
        REASEARCH.check_normal(df_valid[{'bias_XGB'}], output_tag='bias_XGB')
        REASEARCH.check_normal(df_valid[{'bias_APACADA'}], output_tag='bias_APACADA')

        # hourly validation:
        df_valid.plot(kind='')
        df_valid[{'bias_XGB', 'bias_APACADA'}].groupby(df_valid.index.hour).mean().plot(kind='bar')
        plt.title(f'hourly mean bias')
        plt.savefig(f'./plot/bias_in_hour.png', dpi=300)
        plt.show()

    if cfg.job.valid.by_height:
        # to prepare ct, and merge it to the df_valid:
        # ct = pd.read_pickle(cfg.file.ct_SAF_NWC_moufia)
        # REASEARCH.add_ct_for_validation(df_valid, ct)
        ct = pd.read_pickle(cfg.file.data_valid_ct)

        # limit the time difference between lacy obs and saf_nwc data:
        ct2 = ct[ct.diff_minute < 8]
        print(f'{len(ct2) / len(ct) * 100: 4.2f}% are with in 8 minutes')

        # prepare data for violin:
        df3 = pd.DataFrame(pd.concat([ct2['ct'], ct2['ct']]))
        df3.insert(0, 'bias', np.concatenate([ct2['bias_XGB'].values, ct2['bias_APACADA'].values]))
        df3.insert(0, 'method', ['XGBoost' for x in range(len(ct2))] + ['APACADA' for i in range(len(ct2))])

        GEO_PLOT.plot_violin_df_1D(df=df3, x='ct', y='bias', y_unit='unitless', x_label='cloud types', hue='method',
                                   y_label='bias vs LACy CF', split=False, scale='area', inner='box', add_number=True,
                                   x_ticks_labels=['clearsky', 'vLow', 'Low', 'Med', 'HOp', 'vHOp', 'sTp'],
                                   suptitle_add_word='test')
        del df3

        print(f'good')

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.data))):
        print('start to work...')


if __name__ == "__main__":
    sys.exit(cloud())
