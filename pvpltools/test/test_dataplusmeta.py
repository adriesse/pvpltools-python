# -*- coding: utf-8 -*-
"""
@author: Anton Driesse, PV Performance Labs
"""

import os
import numpy as np
import pandas as pd

import pytest
from pandas.testing import assert_frame_equal

from pvpltools.dataplusmeta import DataPlusMeta

#%%

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, 'data')

test_file = os.path.join(test_dir, 'data', 'sample_iec.csv')
test_file_copy = os.path.join(test_dir, 'data', 'sample_iec_copy.csv')
test_file_data = os.path.join(test_dir, 'data', 'sample_iec_copy2.csv')

#%%

def test_1():
    pass

#%%

    # read a file
    dpm = DataPlusMeta.from_txt(test_file)

    # write a copy
    dpm.to_txt(test_file_copy)

    # read the copy
    dpm2 = DataPlusMeta.from_txt(test_file_copy)

    # compare with the original
    assert np.all(dpm.meta == dpm2.meta)
    assert_frame_equal(dpm.cdef, dpm2.cdef)
    assert_frame_equal(dpm.data, dpm2.data)

    # read without using dtypes
    with pytest.warns(UserWarning, match='dtypes'):
        dpm = DataPlusMeta.from_txt(test_file, use_dtypes=False)

    with pytest.raises(AssertionError, match='Attribute "dtype" are different'):
        assert_frame_equal(dpm.data, dpm2.data)

#%%

def test_2():
    pass

#%%

    # see what happens when I start with only data and no cdef

    dpm = DataPlusMeta.from_txt(test_file)
    df = dpm.data

    with pytest.warns(UserWarning, match='Either cdef or data is missing.'):
        dpm3 = DataPlusMeta(data=df)

    with pytest.warns(UserWarning, match='Either cdef or data is missing.'):
        assert not dpm3.check_cdef(raise_on_mismatch=False)

    with pytest.raises(RuntimeError, match='Either cdef or data is missing.'):
        assert not dpm3.check_cdef(raise_on_mismatch=True)

    with pytest.raises(RuntimeError, match='Either cdef or data is missing.'):
        dpm3.to_txt(test_file_data, update_cdef=False)

    assert dpm3.update_cdef()
    assert dpm3.check_cdef()
    dpm3.to_txt(test_file_data, update_cdef=False)

    # once again with fewer steps
    with pytest.warns(UserWarning, match='Either cdef or data is missing.'):
        dpm3 = DataPlusMeta(data=df)

    dpm3.to_txt(test_file_data, update_cdef=True)

#%%

def test_3():
    pass

#%%

    # test a few different cdef vs data inconsistencies

    dpm = DataPlusMeta.from_txt(test_file)

    dpm.data.rename(columns={'Date':'DATE'}, inplace=True)
    with pytest.warns(UserWarning, match='Labels'):
        assert not dpm.check_cdef()

    dpm.data.rename(columns={'DATE':'Date'}, inplace=True)
    assert dpm.check_cdef()

    dpm.data['G'] = dpm.data['G'].astype('float')
    with pytest.warns(UserWarning, match='dtypes'):
        assert not dpm.check_cdef()

    dpm.cdef = None
    with pytest.warns(UserWarning, match='missing'):
        assert not dpm.check_cdef()

    dpm.data = None
    assert dpm.check_cdef()

    with pytest.raises(RuntimeError, match='no data'):
        dpm.to_txt('no file')

#%%

if __name__ == '__main__':

    test_1()
    test_2()
    test_3()
