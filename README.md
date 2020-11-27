# PV Performance Labs Tools for Python

Useful tools for photovoltaics and beyond!

Latest news
-----------

2020-11-27: Six models and a model fitting function are now available, and the example code is coming along nicely too


Contents
--------

A quick overview:

- pvpltools/
	- module_efficiency.py (work in progress)
		- a collection of models for PV module efficiency (at MPP)
		- includes the new ADR model and others
        - also includes a model fitting function
        - demonstrations in a Jupyter Notebook in examples directory
	- iec61853.py
		- reliable functions for Climate-Specific Energy Rating (CSER) calculations
		- incident angle modifier for direct and diffuse irradiance
		- spectral correction/mismatch factor
		- module operating temperature
		- efficiency matrix interpolation/extrapolation

	- dataplusmeta.py
		- a simple way to pack data and essential meta-data into a single text file

- data/
	- nrel_mpert/
		- module measurements, model parameters and other data in DataPlusMeta style

- examples/
    - module_efficiency_demo.ipynb
    - dataplusmeta_demo.py


Development status
------------------

2020-11-27
- The new module `module_efficiency.py` now contains functions for:
    - the new ADR model
    - HEY
    - MotherPV
    - PVGIS
    - MPM5
    - MPM6
    - fitting any of the models to measurements

2020-02-29

- The main building blocks for the Energy Rating calculation are complete.
- A higher level function of example script may be coming some day.


Copyright
---------

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.


License
-------

GPL-3.0, but feel free to let me know if that causes any problems!


Citing
------

When referring to this software in either academic or commercial context,
please use a citation similar to the following:

- A. Driesse,
"PV Performance Labs Tools for Python", (2020), GitHub repository,  https://github.com/adriesse/pvpltools-python

When referring to specific functions, docs strings or algorithms,
please add specifics to the citation.

The following report introduces the new ADR PV module efficiency model
and compares it to the IEC 61853 efficiency matrix interpolation/extrapolation method as well as several other published models:

- Driesse, Anton, & Stein, Joshua.
  "From IEC 61853 power measurements to PV system simulations."
  SAND2020-3877, Sandia National Laboratories, Albuquerque, NM, 2020. [doi:10.2172/1615179.][102]

[102]: https://pvpmc.sandia.gov/download/7737/

Additional publications related to the contents of pvpltools-python
will be listed here as they become available.


Acknowledgements
----------------

The contents of this repository have been developed
before, during or after various projects; as a product or byproduct;
with funding in whole, in part, or not at all.

I would like to acknowledge Sandia National Labs and the US DOE for
substantial project funding as well encouragement to publish open source code.

I also acknowledge and thank all the contributors to pvlib-python,
parts of which I use very frequently in the context of my work.


Getting help
------------

I expect that most potential users will have sufficient knowledge of github and python
that they will be able successfully clone or download, install and `import pvpltools`.

If that's not you, then:

 - click on the green `Clone or download` button and download everything as a zip file
 - unzip the contents somewhere on your computer
 - add the directory called `pvpltools-python-master` to your python search path
 - try `import pvpltools`

The doc strings within the code are currently the primary source of documentation.

Feel free to contact me with questions or suggestions though.
For commercial use, extended support and related services are available.
