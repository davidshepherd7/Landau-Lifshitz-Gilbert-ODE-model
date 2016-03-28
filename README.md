Python LLG ODE solver for experimenting with timesteppers
========================================================

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


If you're using this code please let me know! At the moment I occasionally make sweeping changes to interfaces under the assumption that I'm the sole user. So tell me and I'll try not to break your code. My email address is david[rest of my username] [at] gmail [dot] com.
