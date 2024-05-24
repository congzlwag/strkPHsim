# Streaking Simulation of Photoelectron from Direct Ionization
Function `PHInt_Pspace` in `streakPH.jl` computes the (unnormalized) distribution density of streaked photoelectron based on strong-field approximation. 
Detailed derivations are in [Li et al Opt. Express 26, 4, 4531 (2018)](https://doi.org/10.1364/OE.26.004531).
It uses multi-threading to parallelize over the momentum grid points. 


## Demo
A working example is `demo.jl`. To run it:
```
julia -t [nthread] -i demo.jl
```
Here `[nthread]` is the number of threads, which is usually chosen according to the machine spec. 
The interactive option `-i` allows the visualization to stay as the script is finished. 
If no visualization is requested (`PLOT=false`) then `-i` is unnecessary. 

The output is a hdf5 file specified by variable `out_h5path`.
The control parameters for this demo is gathered under the comment line `#Main input parameters`. 

## Dependencies

* julia >= 1.9.0
* `streakPH.jl` and `utilsv2.jl` in this repo, and the packages imported therein.
