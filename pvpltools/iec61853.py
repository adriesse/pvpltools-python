"""
This module implements the calculations energy rating calculations
described in standard IEC-61853.

The four main calculation steps are:

    1. Calculate the angle-of-incidence correction using the functions:
        - martin_ruiz()
        - martin_ruiz_diffuse()
    2. Evaluate the spectral factor using the functions:
        - convert_to_banded()
        - calc_spectral_factor()
    3. Estimate the operating temperature using the function:
        - faiman()
    4. Determine the module efficiency using the class:
        - BilinearInterpolator()

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.
"""

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

__all__ = [
    'convert_to_banded',
    'calc_spectral_factor',
    'BilinearInterpolator',
    'martin_ruiz',
    'martin_ruiz_diffuse',
    'faiman',
    ]

# This constant defines the fixed boundaries between the 29 spectral bands
# used in the IEC 61853-4 climate profiles.

SPECTRAL_BAND_EDGES = (
     306.8,  327.8,  362.5,  407.5,  452.0,  517.7,  540.0,  549.5,
     566.6,  605.0,  625.0,  666.7,  684.2,  704.4,  742.6,  791.5,
     844.5,  889.0,  974.9, 1045.7, 1194.2, 1515.9, 1613.5, 1964.8,
    2153.5, 2275.2, 3001.9, 3635.4, 3991.0, 4605.65
    )

# This constant defines irradiance of the AM15G spectrum in each of the 29
# spectral bands used in the IEC 61853-4 climate profiles.
# NB: the value in the last band represents only the range 3991-4000nm.

BANDED_AM15G = (
     3.69618,  16.60690,  34.22227,  56.13480, 101.36658,  33.98919,
    14.45719,  25.91979,  56.57334,  29.05096,  58.35457,  24.50170,
    25.22513,  45.09050,  52.90410,  52.42015,  42.28474,  47.29518,
    49.45154,  61.54425,  67.93106,  24.83419,  34.04581,  13.98155,
     9.11025,   9.07606,   4.22986,   3.04112,   0.06470
    )

#%%

def convert_to_banded(spectral_reponse):
    """
    Calculate the mean spectral reponse in standard wavelength bands.

    The mean value for each band is calculated as the area under the
    linearly interpolated spectral response (SR) curve between band edges,
    divided by the width of the band.  The band edges are defined in
    IEC 61853-4. [1]

    Parameters
    ----------
    spectral_reponse : pandas.Series
        Spectral response.  The index of spectral_reponse must contain
        wavelengths [nm]; the values in spectral_reponse may be
        relative or absolute.

    Returns
    -------
    np.array
        Mean values of the spectral_reponse in the specified wavelength bands.

    Notes:
    ------
    - The trapezoid method is used to calculate the area under the SR curve.
    - The SR at the band edges is estimated by linear interpolation.
    - Negative values in the SR are replaced with zero.

    References
    ----------

    .. [1] "IEC 61853-4 Photovoltaic (PV) module performance testing and
       energy rating - Part 4: Standard reference climatic profiles".
       IEC, Geneva, 2018.

    Author: Anton Driesse, PV Performance Labs
    """
    sr = spectral_reponse

    band_edges = SPECTRAL_BAND_EDGES

    # remember these limits for calculating the areas
    sr_left = sr.index.min()
    sr_right = sr.index.max()

    # insert extra points into the SR at the band edges
    extra_wavelengths = set(band_edges) - set(sr.index)
    sr = sr.append(pd.Series(np.nan, extra_wavelengths))
    sr = sr.sort_index()
    sr = sr.clip(0.0)
    sr = sr.interpolate(method='index', limit_area='inside')
    sr = sr.fillna(0.0)

    # calculate the mean value of the SR within each band
    band_means = []

    for band_left, band_right in zip(band_edges[:-1], band_edges[1:]):

        limit_left = max(sr_left, band_left)
        limit_right = min(sr_right, band_right)
        band = sr.loc[limit_left:limit_right]

        area = np.trapz(band, band.index)
        width = band_right - band_left
        band_means.append(area / width)

    return np.array(band_means)


def calc_spectral_factor(banded_irradiance, banded_responsivity):
    """
    Calculate the spectral correction/mismatch factor(s) for the given
    spectral irradiance and device responsivity.

    Parameters
    ----------
    banded_irradiance : 1-D or 2-D array_like
        Spectral irradiance integrated in 29 bands [W/m²].

    banded_responsivity : 1-D array_like
        Spectral responsivity averaged in 29 bands [unitless].

    Raises
    ------
    ValueError
        If the number of bands in the arguments is incorrect.

    Returns
    -------
    scalar or 1-D array
        The calculated spectral correction factors [unitless].

    See also
    --------
    SPECTRAL_BAND_EDGES, BANDED_AM15G, convert_to_banded

    Notes
    -----
    The calculation method used here does not correspond precisely to the
    description in IEC 61853-3 [1] because the latter has inconsistencies.
    In particular:

    - the standard specifies integration limits of 300 and 4000 nm
    - the outer edges of the spectral bands in the standard's climate profiles
      are 306.8 and 4605.65 nm
    - the AM15G reference spectrum has irradiance values from 280 to 4000 nm
    - the AM15G reference spectrum must be integrated from 280 nm to
      infinity to reach the value of 1000 W/m2

    Spectral mismatch calculations require consistency in the limits.
    In this function all integration limits are set to the band edges
    306.8 and 3991, and the fixed value of 1000 W/m2 is replaced with the
    integral of the AM15 spectrum within these limits.  This implies
    that the last spectral band in the climate files is not used.

    The maximum difference in spectral factor between this simplification
    and multiple options for imposing a 4000 nm upper limit was
    found to be < 80 ppm.

    References
    ----------
    .. [1] "IEC 61853-3 Photovoltaic (PV) module performance testing and energy
       rating - Part 3: Energy rating of PV modules". IEC, Geneva, 2018.

    Author: Anton Driesse, PV Performance Labs
    """

    banded_irradiance = np.asanyarray(banded_irradiance)
    banded_responsivity = np.asanyarray(banded_responsivity)

    if (banded_irradiance.ndim < 1) or (banded_irradiance.shape[-1] != 29):
        raise ValueError('banded_irradiance must 29 bands')

    if (banded_responsivity.ndim != 1) or (banded_responsivity.shape[0] != 29):
        raise ValueError('banded_responsivity must have 29 bands')

    # the mask is used to ignore the last band in all the spectra
    mask = np.array( 28 * [1] + [0])

    with np.errstate(invalid='ignore'):
        eta_real = np.sum(mask * banded_irradiance * banded_responsivity, -1) \
                 / np.sum(mask * banded_irradiance, -1)

        eta_am15 = np.sum(mask * BANDED_AM15G * banded_responsivity) \
                 / np.sum(mask * BANDED_AM15G)

    spectral_factor = eta_real / eta_am15

    return spectral_factor


#%%

class BilinearInterpolator(RegularGridInterpolator):
    """
    Bilinear interpolation and extrapolation of a matrix of values.

    The values in the matrix are measurements at various combinations
    of irradiance and temperature. The matrix may be completely filled,
    or there may be missing values at high irradiance/low temperature,
    or at low irradiance/high temperature combinations.  These are filled in
    using the method described in [1], which ensures a continuous
    interpolation/extrapolation surface.

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

    Author: Anton Driesse, PV Performance Labs
    """

    def __init__(self, matrix):

        m = matrix.sort_index(0).sort_index(1)

        num_iterations = max(m.shape) - 1

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

    Author: Anton Driesse, PV Performance Labs
    '''

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


def martin_ruiz_diffuse(surface_tilt, a_r=0.16, c1=None, c2=None):
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
        If c1 is None, the constant 4 / 3 / pi (~0.4244) is used [1]_.

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

    Author: Anton Driesse, PV Performance Labs
    '''

    if isinstance(surface_tilt, pd.Series):
        out_index = surface_tilt.index
    else:
        out_index = None

    surface_tilt = np.asanyarray(surface_tilt)

    # avoid undefined results for horizontal or upside-down surfaces
    zeroang = 1e-06

    surface_tilt = np.where(surface_tilt == 0, zeroang, surface_tilt)
    surface_tilt = np.where(surface_tilt == 180, 180 - zeroang, surface_tilt)

    if c1 is None:
        # This value is from [1]
        c1 = 4 / 3 / np.pi

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

    Author: Anton Driesse, PV Performance Labs
    '''

    # The following lines may seem odd since u0 & u1 are probably scalar,
    # but it serves an indirect and easy way of allowing lists and
    # tuples for the other function arguments.
    u0 = np.asanyarray(u0)
    u1 = np.asanyarray(u1)

    total_loss_factor = u0 + u1 * wind_speed
    heat_input = poa_global
    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference

