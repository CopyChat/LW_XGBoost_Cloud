"""
to apply ML method to improve cloud fraction estimation,
compared with traditional physical cloud method based on thermal physics,
where assumptions made in lower level of atmosphere.
"""

__version__ = f'Version 2.0  \nTime-stamp: <2022-08-19>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
# import subprocess
# import numpy as np
# import pandas as pd
# import xarray as xr
# from importlib import reload
# import matplotlib.pyplot as plt
from omegaconf import DictConfig

import GEO_PLOT
import REASEARCH
import PUBLISH


@hydra.main(config_path="configs", config_name="config")
def cloud(cfg: DictConfig) -> None:
    """
    """
    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.data))):
        print('start to work...')


if __name__ == "__main__":
    sys.exit(cloud())
