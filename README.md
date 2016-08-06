Becca is a general learning program for use in any robot or embodied system.
When using Becca, a robot learns to do whatever it is rewarded to do, 
and continues learning throughout its lifetime. This package contains 
some test worlds and helps to run them. This helps ensure that 
Becca is working as intended.

####Install the Becca test worlds.

From the command line:

    pip install becca_test

`becca` installs automatically when you install `becca_test`. 

####Run Becca on a decathlon of test worlds.

    python
    >>>import becca_test.test
    >>>becca_test.test.suite()
    
####Run a single world.

To run the test world `grid_1D.py` from the command line:

    python -m grid_1D

<a href="url"><img src="https://github.com/brohrer/becca-docs/raw/master/figs/logo_plate.png" 
align="center" height="40" width="120" ></a>
 
