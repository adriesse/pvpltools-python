"""
Copyright 2020 Anton Driesse (@adriesse), PV Performance Labs.

@author: Anton Driesse
"""

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal

from pvpltools import iec61853

#%%

def test_BilinearInterpolator():

    # The following are tested:

    # Filling completed
    # Filled value
    # Interpolated value
    # Extrapolated value

    eta = pd.DataFrame(index=[1000, 1100],
                       columns=[15, 25],
                       data=[[  22.0, 20.0],
                              [np.nan, 19.0]])

    interpolate_eta = iec61853.BilinearInterpolator(eta)
    assert not np.any(np.isnan(interpolate_eta.values))

    result = interpolate_eta([1100, 1000, 900], [15, 20, 25])
    assert_allclose(result, [21, 21, 21])

