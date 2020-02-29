"""
Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.
"""

import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_allclose, assert_equal

from pvpltools.iec61853 import SPECTRAL_BAND_EDGES, BANDED_AM15G
from pvpltools.iec61853 import (convert_to_banded, calc_spectral_factor,
                                BilinearInterpolator,
                                martin_ruiz, martin_ruiz_diffuse,
                                faiman)

#%%

def test_convert_to_banded():

    # simple sr with high plateau
    sr = pd.Series(index=[280.0, 889.0, 1000.0, 1150.0],
                   data=[   0.0,   1.0,    1.0,    0.0])

    sr_k = convert_to_banded(sr)
    # at least one band is in the plateau
    assert_equal(np.max(sr_k), 1.0)
    assert_equal(np.count_nonzero(sr_k), 20)
    assert_allclose(np.mean(sr_k), 0.36986736)

#%%

def test_calc_spectral_factor():

    # make three test spectra
    bi0 = np.array(BANDED_AM15G)
    bi1 = bi0 * np.linspace(1.1, 0.5, 29) # blue enhanced
    bi2 = bi0 * np.linspace(0.5, 1.1, 29) # red enhanced

    bi = np.vstack([bi0, bi1, bi2])

    # flat SR beyond limts
    sr = pd.Series([1.0, 1.0], [200, 5000])
    bsr = convert_to_banded(sr)
    smm = calc_spectral_factor(bi, bsr)
    assert_equal(smm, [1, 1, 1])

    # flat SR exactly to limits
    sr = pd.Series([1.0, 1.0], [SPECTRAL_BAND_EDGES[0], SPECTRAL_BAND_EDGES[-2]])
    bsr = convert_to_banded(sr)
    smm = calc_spectral_factor(bi, bsr)
    assert_equal(smm, [1, 1, 1])

    # flat SR in Si range
    sr = pd.Series([1.0, 1.0], [300, 1200])
    bsr = convert_to_banded(sr)
    smm = calc_spectral_factor(bi, bsr)
    assert_allclose(smm, [1., 1.04744717, 0.94786063])

    # sawtooth SR in Si range
    sr = pd.Series([0.1, 1.0, 0.0], [300, 1000, 1200])
    bsr = convert_to_banded(sr)
    smm = calc_spectral_factor(bi, bsr)
    assert_allclose(smm, [1., 1.00059311, 0.99934824])

    # scaling doesn't make a difference
    smm = calc_spectral_factor(bi, bsr * 3)
    assert_allclose(smm, [1., 1.00059311, 0.99934824])

    smm = calc_spectral_factor(bi * 2, bsr)
    assert_allclose(smm, [1., 1.00059311, 0.99934824])

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

    interpolate_eta = BilinearInterpolator(eta)
    assert not np.any(np.isnan(interpolate_eta.values))

    result = interpolate_eta([1100, 1000, 900], [15, 20, 25])
    assert_allclose(result, [21, 21, 21])

