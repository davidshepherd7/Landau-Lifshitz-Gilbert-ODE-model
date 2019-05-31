Python LLG ODE solver for experimenting with timesteppers
========================================================

**Note**: This code is essentially **unmaintained**, it was written for experiments as part of my PhD and I have since moved on to other things. Here's an explaination I wrote (elsewhere) of when you might want to use this code and what the alternatives are:

> The LLG equation is usually a PDE but it can be simplified to an ODE when all of the fields making up H_eff are constant in space. Physically this happens when you have a small (relative to the exchange length) ellipsoid of material. There's a bit more explanation about it [my thesis](https://www.escholar.manchester.ac.uk/uk-ac-man-scw:266267) in section 7.4.1. The code in this repository only deals with the ODE case. This is a useful test case if you are interested in experimenting with time integration methods, e.g. [the adaptive implicit midpoint rule that I was experimenting with](https://link.springer.com/article/10.1007/s10915-019-00965-8). It is potentially useful for some *very simple* physical problems.

> For most physical problems you will need code that also models variations in space. The most widely used software for this is [OOMMF](http://math.nist.gov/oommf/) which uses finite differences for the spatial part. There's also [nmag](http://nmag.soton.ac.uk/nmag/) which uses finite element methods instead. Finite element methods allow you to accurately model more complex shapes (i.e. shapes that aren't cubeoids), but the underlying maths can be more difficult to understand. I used [oomph-lib](http://oomph-lib.maths.man.ac.uk/doc/html/index.html) together with [some extensions for micromagnetics](https://github.com/davidshepherd7/oomph-lib-micromagnetics), however the micromagnetics extensions are experimental and unmaintained so you probably only want to use this if you are directly following up on my research.


Setup
--------

Required python modules: scipy, matplotlib, sympy.

Clone into a directory named simpleode with the command:

    git clone https://github.com/davidshepherd7/Landau-Lifshitz-Gilbert-ODE-model.git simpleode
    
Note that the name of the directory *must* be simpleode for the import statements to work. Next add the directory *containing* the simpleode directory to your python path with:

    export PYTHONPATH="$PYTHONPATH:[path/to/simpleode/../]"
    
you probably want to put this in your shell rc file so that it is set permenantly.

Apologies for the convoluted setup, I wrote this code when I was fairly new to python and I didn't follow proper packaging practices.

Testing
---------

Run self tests with

    nosetests --all-modules
    
in the simplellg directory (this requires the nose package:

    sudo apt-get install python-nose

on Debian based systems).
