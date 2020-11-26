"""
This module contains implementations of various PV module efficiency models.
These models have a common purpose, which is to predict the efficiency as
maximum power point as a function of operating conditions--mainly irradiance
and temperature, but wind could also be included.

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.
"""

import numpy as np

def adr(irradiance, temperature,
        k_a, k_d, tc_d, k_rs, k_rsh):
    '''
    Calculate PV module efficiency using the ADR model

    The efficiency varies with irradiance and operating temperature
    and is determined by 5 model parameters [1]_.

    Parameters
    ----------
    irradiance : non-negative numeric, W/m²
        The effective irradiance incident on the PV module.

    temperature : numeric, °C
        The module operating temperature.

    k_a : float
        Absolute scaling factor, which is equal to the efficiency at
        reference conditions. This factor allows the model to be used
        with relative or absolute efficiencies, and to accommodate data sets
        which are not perfectly normalized but have a slight bias at
        the reference conditions.

    k_d : negative float
        “Dark irradiance” or diode coefficient which influences the voltage
        increase with irradiance.

    tc_d : float
        Temperature coefficient of the diode coefficient, which indirectly
        influences voltage. Because it is the only temperature coefficient
        in the model, its value will also reflect secondary temperature
        dependencies that are present in the PV module.

    k_rs and k_rsh : float
        Series and shunt resistance loss factors. Because of the normalization
        they can be read as power loss fractions at reference conditions.
        For example, if k_rs is 0.05, the internal loss assigned to the
        series resistance has a magnitude equal to 5% of the module output.

    Returns
    -------
    eta : numeric
        The efficiency of the module at the specified irradiance and
        temperature.

    Notes
    -----
    The efficiency values may be absolute or relative, and may be expressed
    as percent or per unit.  This is determined by the efficiency data
    used to derive values for the 5 model parameters.  The first model
    parameter k_a is equal to the efficiency at STC and therefore
    indicates the efficiency scale being used.  k_a can also be changed
    freely to adjust the scale, or to change the module class to a slightly
    higher or lower efficiency.

    References
    ----------
    .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
       to PV system simulations", Sandia Report No. SAND2020-3877, 2020.

    .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic
       Module Efficiency Model for Energy Prediction and Rating",
       forthcoming.

    Author: Anton Driesse, PV Performance Labs
    '''
    g = np.asanyarray(irradiance)
    T = np.asanyarray(temperature)

    # normalize the irradiance
    G_REF = 1000
    s = g / G_REF

    # obtain the difference from reference temperature
    T_REF = 25
    dT   = T - T_REF
    t_abs = T + 273.15

    # equation 29 in JPV
    s_o     = 10**(k_d + (tc_d * dT))
    s_o_ref = 10**(k_d)

    # equation 28 and 30 in JPV
    # the constant k_v does not appear here because it cancels out
    v  = np.log(s / s_o     + 1)
    v /= np.log(1 / s_o_ref + 1)

    # equation 25 in JPV
    eta  = k_a * ((1 + k_rs + k_rsh) * v - k_rs * s - k_rsh * v**2)

    return eta

