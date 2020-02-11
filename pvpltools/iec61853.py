"""
This module implements the calculations described in standard IEC-61853.

Three functions that were previously contributed to pvlib-python
are also included here for convenience.

Copyright 2020 Anton Driesse (@adriesse), PV Performance Labs.

@author: Anton Driesse
"""

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

#%%

class BilinearInterpolator(RegularGridInterpolator):
    """
    Bilinear interpolation and extrapolation of a matrix of values
    as

    The values in the matrix are measurements at various combinations
    of irradiance and temperature. The matrix may be completely filled,
    or there may be missing (nan) values at high irradiance/low temperature,
    or at low irradiance/high temperature combinations.  These are filled in
    using the method described in [1], which ensures a continuous
    interpolation/extrapolation surface.  The remaining

    Parameters
    ----------
    matrix : pandas.DataFrame

        The row index of matrix must contain irradiance values and
        the column index of matrix must contain temperature values.
        The data in the dataframe can represent anything, but will
        usually be PV module efficiency.  This can be relative or
        absolute, as fraction or percent.

    Methods
    -------
    __call__

    Notes
    -----

    This class is implemented as a subclass of
    scipy.interpolate.RegularGridInterpolator.  Essentially it ensures the
    regular grid is completely filled, as required, and that the appropriate
    options are selected.  It also provides some domain-specific documentation.

    Example
    -------

    >>> eta = pd.DataFrame(index=[1000, 1100],
    ...                    columns=[15, 25],
    ...                    data=[[  22.0, 20.0],
    ...                          [np.nan, 19.0]])
    >>> eta
            15    25
    1000  22.0  20.0
    1100   NaN  19.0
    >>> interpolate_eta = BilinearInterpolator(eta)
    >>> interpolate_eta.values
    array([[22., 20.],
           [21., 19.]])
    >>> interpolate_eta([900, 1100], [25, 15])
    array([21., 21.])

    See also
    --------
    scipy.interpolate.RegularGridInterpolator

    References
    ----------
    .. [1] Anton Driesse "Report title" SAND-20xx forthcoming
    .. [3] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    """

    def __init__(self, matrix):

        m = matrix.sort_index(0).sort_index(1)

        num_iterations = min(m.shape) - 1

        for i in range(num_iterations):
            # extrapolate and fill toward upper right corner of matrix
            m.fillna(m.shift(+1, axis=1) + m.shift(-1, axis=0) -
                     m.shift(-1, axis=0).shift(+1, axis=1), inplace=True)

            # extrapolate and fill toward lower left corner of matrix
            m.fillna(m.shift(-1, axis=1) + m.shift(+1, axis=0) -
                     m.shift(+1, axis=0).shift(-1, axis=1), inplace=True)

        return super().__init__((m.index, m.columns), m.values,
                                 method='linear',
                                 bounds_error=False,
                                 fill_value=None)

    def __call__(self, irradiance, temperature):
        '''
        Interpolate/extrapolate at specified conditions

        Parameters
        ----------
        irradiance, temperature : array_like
            The conditions to be interpolated/extrapolated from the matrix.

        '''
        return super().__call__((irradiance, temperature))

#%%

def martin_ruiz(aoi, a_r=0.16):
    r'''
    Determine the incidence angle modifier (IAM) using the Martin
    and Ruiz incident angle model.

    Parameters
    ----------
    aoi : numeric, degrees
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees.

    a_r : numeric
        The angular losses coefficient described in equation 3 of [1]_.
        This is an empirical dimensionless parameter. Values of ``a_r`` are
        generally on the order of 0.08 to 0.25 for flat-plate PV modules.

    Returns
    -------
    iam : numeric
        The incident angle modifier(s)

    Notes
    -----
    `martin_ruiz` calculates the incidence angle modifier (IAM) as described in
    [1]_. The information required is the incident angle (AOI) and the angular
    losses coefficient (a_r). Note that [1]_ has a corrigendum [2]_ which
    clarifies a mix-up of 'alpha's and 'a's in the former.

    The incident angle modifier is defined as

    .. math::

       IAM = \frac{1 - \exp(-\cos(\frac{aoi}{a_r}))}
       {1 - \exp(\frac{-1}{a_r}}

    which is presented as :math:`AL(\alpha) = 1 - IAM` in equation 4 of [1]_,
    with :math:`\alpha` representing the angle of incidence AOI. Thus IAM = 1
    at AOI = 0, and IAM = 0 at AOI = 90.  This equation is only valid for
    -90 <= aoi <= 90, therefore `iam` is constrained to 0.0 outside this
    interval.

    References
    ----------
    .. [1] N. Martin and J. M. Ruiz, "Calculation of the PV modules angular
       losses under field conditions by means of an analytical model", Solar
       Energy Materials & Solar Cells, vol. 70, pp. 25-38, 2001.

    .. [2] N. Martin and J. M. Ruiz, "Corrigendum to 'Calculation of the PV
       modules angular losses under field conditions by means of an
       analytical model'", Solar Energy Materials & Solar Cells, vol. 110,
       pp. 154, 2013.

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. July, 2019

    aoi_input = aoi

    aoi = np.asanyarray(aoi)
    a_r = np.asanyarray(a_r)

    if np.any(np.less_equal(a_r, 0)):
        raise ValueError("The parameter 'a_r' cannot be zero or negative.")

    from numpy import cos, radians

    with np.errstate(invalid='ignore'):
        iam = (1 - np.exp(-cos(radians(aoi)) / a_r)) / (1 - np.exp(-1 / a_r))
        iam = np.where(np.abs(aoi) >= 90.0, 0.0, iam)

    if isinstance(aoi_input, pd.Series):
        iam = pd.Series(iam, index=aoi_input.index)

    return iam


def martin_ruiz_diffuse(surface_tilt, a_r=0.16, c1=0.4244, c2=None):
    '''
    Determine the incidence angle modifiers (iam) for diffuse sky and
    ground-reflected irradiance using the Martin and Ruiz incident angle model.

    Parameters
    ----------
    surface_tilt: float or array-like, default 0
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
        surface_tilt must be in the range [0, 180]

    a_r : numeric
        The angular losses coefficient described in equation 3 of [1]_.
        This is an empirical dimensionless parameter. Values of a_r are
        generally on the order of 0.08 to 0.25 for flat-plate PV modules.
        a_r must be greater than zero.

    c1 : float
        First fitting parameter for the expressions that approximate the
        integral of diffuse irradiance coming from different directions.
        c1 is given as the constant 4 / 3 / pi (0.4244) in [1]_.

    c2 : float
        Second fitting parameter for the expressions that approximate the
        integral of diffuse irradiance coming from different directions.
        If c2 is None, it will be calculated according to the linear
        relationship given in [3]_.

    Returns
    -------
    iam_sky : numeric
        The incident angle modifier for sky diffuse

    iam_ground : numeric
        The incident angle modifier for ground-reflected diffuse

    Notes
    -----
    Sky and ground modifiers are complementary: iam_sky for tilt = 30 is
    equal to iam_ground for tilt = 180 - 30.  For vertical surfaces,
    tilt = 90, the two factors are equal.

    References
    ----------
    .. [1] N. Martin and J. M. Ruiz, "Calculation of the PV modules angular
       losses under field conditions by means of an analytical model", Solar
       Energy Materials & Solar Cells, vol. 70, pp. 25-38, 2001.

    .. [2] N. Martin and J. M. Ruiz, "Corrigendum to 'Calculation of the PV
       modules angular losses under field conditions by means of an
       analytical model'", Solar Energy Materials & Solar Cells, vol. 110,
       pp. 154, 2013.

    .. [3] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Oct. 2019

    if isinstance(surface_tilt, pd.Series):
        out_index = surface_tilt.index
    else:
        out_index = None

    surface_tilt = np.asanyarray(surface_tilt)

    # avoid undefined results for horizontal or upside-down surfaces
    zeroang = 1e-06

    surface_tilt = np.where(surface_tilt == 0, zeroang, surface_tilt)
    surface_tilt = np.where(surface_tilt == 180, 180 - zeroang, surface_tilt)

    if c2 is None:
        # This equation is from [3] Sect. 7.2
        c2 = 0.5 * a_r - 0.154

    beta = np.radians(surface_tilt)

    from numpy import pi, sin, cos, exp

    # because sin(pi) isn't exactly zero
    sin_beta = np.where(surface_tilt < 90, sin(beta), sin(pi - beta))

    trig_term_sky = sin_beta + (pi - beta - sin_beta) / (1 + cos(beta))
    trig_term_gnd = sin_beta +      (beta - sin_beta) / (1 - cos(beta)) # noqa: E222 E261 E501

    iam_sky = 1 - exp(-(c1 + c2 * trig_term_sky) * trig_term_sky / a_r)
    iam_gnd = 1 - exp(-(c1 + c2 * trig_term_gnd) * trig_term_gnd / a_r)

    if out_index is not None:
        iam_sky = pd.Series(iam_sky, index=out_index, name='iam_sky')
        iam_gnd = pd.Series(iam_gnd, index=out_index, name='iam_ground')

    return iam_sky, iam_gnd


def faiman(poa_global, temp_air, wind_speed=1.0, u0=25.0, u1=6.84):
    '''
    Calculate cell or module temperature using an empirical heat loss factor
    model as proposed by Faiman [1] and adopted in the IEC 61853
    standards [2] and [3].

    Usage of this model in the IEC 61853 standard does not distinguish
    between cell and module temperature.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance [W/m^2].

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/s is the wind
        speed at module height used to determine NOCT. [m/s]

    u0 : numeric, default 25.0
        Combined heat loss factor coefficient. The default value is one
        determined by Faiman for 7 silicon modules. [W/(m^2 C)].

    u1 : numeric, default 6.84
        Combined heat loss factor influenced by wind. The default value is one
        determined by Faiman for 7 silicon modules. [(W/m^2 C)(m/s)].

    Returns
    -------
    numeric, values in degrees Celsius

    Notes
    -----
    All arguments may be scalars or vectors. If multiple arguments
    are vectors they must be the same length.

    References
    ----------
    [1] Faiman, D. (2008). "Assessing the outdoor operating temperature of
    photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.

    [2] "IEC 61853-2 Photovoltaic (PV) module performance testing and energy
    rating - Part 2: Spectral responsivity, incidence angle and module
    operating temperature measurements". IEC, Geneva, 2018.

    [3] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
    rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Dec., 2019

    # The following lines may seem odd since u0 & u1 are probably scalar,
    # but it serves an indirect and easy way of allowing lists and
    # tuples for the other function arguments.
    u0 = np.asanyarray(u0)
    u1 = np.asanyarray(u1)

    total_loss_factor = u0 + u1 * wind_speed
    heat_input = poa_global
    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference
