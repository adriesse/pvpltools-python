"""
This module provides useful functions related to power conversion for
PV systems, including a flexible implementation of the ADR power loss
model, aka ADR inverter model.

Copyright (c) 2021 Anton Driesse, PV Performance Labs.
"""

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from pvlib.inverter import _sandia_eff

#%%

def adr_converter_core(v_norm, p_norm, b):
    '''
    Calculate power loss in a power converter for the given normalized
    operating conditions and the ADR model "b" coefficients.

    The ADR model input and output quantities are normalized using a nominal
    voltage and power level, allowing the converter characteristics to be
    scaled up or down easily.

    Voltage and current values may correspond to electrical inputs or outputs,
    and may be DC or AC, hence different types of converters can be modeled.
    However in general a set of coefficients is only valid for a specific type.

    Parameters
    ----------

    v_norm, p_norm : numeric
        Normalized voltage and power, unitless.  These arguments must be
        "broadcastable" to the same shape.

    b : 1-D or 2-D
        Nine model "b" coefficients as described in [1].

    Returns
    -------

    p_loss_norm : numeric
        Normalized power loss, unitless.

    References
    ----------

    .. [1] A. Driesse, "Beyond the Curves: Modeling the Electrical Efficiency
       of Photovoltaic Inverters", 33rd IEEE Photovoltaic Specialist
       Conference (PVSC), June 2008


    Author: Anton Driesse, PV Performance Labs
    '''

    v_norm = np.asanyarray(v_norm)
    p_norm = np.asanyarray(p_norm)

    # ensure the coefficients have a 3x3 shape
    b = np.reshape(b, (3, 3))

    # calculate the voltage terms
    with np.errstate(divide='ignore'):
        v_terms = np.stack([v_norm ** 0, v_norm - 1, 1 / v_norm - 1])
        v_terms[np.isinf(v_terms)] = 0.0

    # calculate the power term coefficients a0, a1, a2
    a = np.einsum('v...,vp->p...', v_terms, b)

    # calculate the power terms and the power losses
    p_terms = np.stack([p_norm ** 0, p_norm, p_norm ** 2])
    p_loss_norm = np.einsum('p...,p...->...', p_terms, a)

    return p_loss_norm


#%%

def fit_adr_converter_core(v_norm, p_norm, p_loss_norm,
                           method='trf', **kwargs):
    '''
    Determine the ADR model "b" coefficients for power converter losses.

    This is a convenience function that calls the scipy `curve_fit` function
    with suitable parameters and defaults.

    Parameters
    ----------

    v_norm, p_norm, p_loss_norm : numeric
        Normalized voltage, power and power loss, unitless. These arguments
        must be "broadcastable" to the same shape.

    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization. See `least_squares` for more details.
        Default is 'trf'.

    kwargs :
        Optional keyword arguments passed to `curve_fit`.

    Returns
    -------
    b : 1-D array
        Nine optimal values for the model "b" coefficients.

    pcov : 2-D array
        The estimated covariance of b. See `curve_fit` for details.


    Author: Anton Driesse, PV Performance Labs
    '''

    def model_wrapper(xdata, *params):
        return adr_converter_core(*xdata, params)

    # broadcast the inputs first, then flatten them for use with curve_fit
    v_n, p_n, p_loss_n = np.broadcast_arrays(v_norm, p_norm, p_loss_norm)

    xdata = np.stack([np.ravel(v_n), np.ravel(p_n)])
    ydata = np.ravel(p_loss_n)

    # the coefficient matrix must also be flat
    b0 = np.array([0.01, 0.01, 0.01,
                   0.00, 0.00, 0.00,
                   0.00, 0.00, 0.00])

    b, pcov = curve_fit(model_wrapper,
                        xdata, ydata,
                        p0=b0,
                        method=method, **kwargs)

    return b.reshape(3, 3), pcov

#%%

def create_cec_matrix_sandia(inverter):
    '''
    Create a matrix of operating points that approximates the cec inverter
    test matrix using the given Sandia inverter model parameters.


    Author: Anton Driesse, PV Performance Labs
    '''
    CEC_LEVELS = [0.1, 0.2, 0.30, 0.50, 0.75, 1.00]

    p_dc = inverter.Pdco * np.array(CEC_LEVELS)
    v_dc = [inverter.Mppt_low, inverter.Vdco, inverter.Mppt_high]

    v_dc, p_dc = np.meshgrid(v_dc, p_dc)

    p_ac = _sandia_eff(v_dc.flatten(), p_dc.flatten(), inverter)

    return v_dc, p_dc, p_ac.reshape(v_dc.shape)

#%%

def fit_adr_to_sandia(snl_params):
    '''
    Create an ADR inverter parameter set that is equivalent to the given
    Sandia inverter model.


    Author: Anton Driesse, PV Performance Labs
    '''
    ADR_KEYS = ['Manufacturer', 'Model', 'Source', 'Vac', 'Vintage',
                'Pacmax', 'Pnom', 'Vnom', 'Vmin', 'Vmax',
                'ADRCoefficients', 'Pnt', 'Vdcmax', 'Idcmax',
                'MPPTLow', 'MPPTHi', 'TambLow', 'TambHi', 'Weight']

    SNL_SOURCES = ['Vac', 'Pnt', 'Vdcmax', 'Idcmax',
                  'Mppt_low', 'Mppt_high', 'Paco',
                  'Pdco',  'Vdco', 'Mppt_low', 'Mppt_high']

    ADR_TARGETS = ['Vac', 'Pnt', 'Vdcmax', 'Idcmax',
                  'MPPTLow', 'MPPTHi', 'Pacmax',
                  'Pnom', 'Vnom', 'Vmin', 'Vmax']

    # create a skeleton Series to receive the ADR parameter values
    adr_params = pd.Series(index=ADR_KEYS, dtype=object)
    adr_params.name = snl_params.name
    adr_params.Manufacturer, adr_params.Model, __ = snl_params.name.split('__')

    # fill in values that are available in the source
    for ks, ka in zip(SNL_SOURCES, ADR_TARGETS):
        adr_params[ka] = snl_params[ks]

    # round off the nominal power
    p_nom = adr_params['Pnom']
    decimals = min(0, 2 - int(np.log10(p_nom) + 0.5))
    adr_params['Pnom'] = np.round(p_nom, decimals)

    # create a set of operating points for fitting the ADR model
    v_dc, p_dc, p_ac = create_cec_matrix_sandia(snl_params)

    # normalize the operating points prior to fitting
    p_loss = (p_dc - p_ac) / adr_params.Pnom
    p_dc = p_dc / adr_params.Pnom
    v_dc = v_dc / adr_params.Vnom

    # fit
    b, __ = fit_adr_converter_core(v_dc, p_dc, p_loss)

    # round off the result
    adr_params['ADRCoefficients'] = b.round(5).ravel().tolist()

    return adr_params
