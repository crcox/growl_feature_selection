This directory contains the workhorse functions for solving the GrOWL optimization.

The implementation is built on top of the SLOPE package, published by M. Bogdan, E. van den Berg, C. Sabatti, W. Su, and E. J. Candès (2015) and distributed under GLP-3.0.

The PDF of this publication is `slope_2015.pdf` within this directory.

The source code is in `SLOPE_code`. It was obtained from https://candes.su.domains/software/SortedL1/, specifically https://candes.su.domains/software/SortedL1/SLOPE_code.tgz. (Many links on the main page are broken, including all links to the code except this one in particular.)

To run GrOWL, only the code in this directory should be on the path. The code in SLOPE_code should NOT be on the path. It is included in case the mex files need to be recompiled.