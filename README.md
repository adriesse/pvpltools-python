# PV Performance Labs Tools for Python

Useful tools for photovoltaics and beyond!

Contents
--------

A quick overview:

- pvpltools/
	- iec61853.py
		Reliable functions for Climate-Specific Energy Rating (CSER) calculations.
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

Development status
------------------

2020-02-29

- The main building blocks for the Energy Rating calculation are complete.
- A higher level function of example script may be coming soon.


Copyright
---------

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.


License
-------

To be confirmed: BSD 3-clause


Citing
------

When referring to this software in either academic or commercial context,
please use a citation similar to the following:

- A. Driesse, "PV Performance Labs Tools for Python", (2020), GitHub repository,
  https://github.com/adriesse/pvpltools-python

When referring to specific functions, docs strings or algorithms,
please add specifics to the citation.

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

I trust that potential users will have sufficient knowledge of github and python
that they will be able successfully download, install and `import pvpltools`.
The doc strings within the code are currently the primary source of documentation.

Feel free to contact me with questions or suggestions though.
For commercial use, extended support and related services are available.
