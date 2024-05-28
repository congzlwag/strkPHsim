using LinearAlgebra
import Statistics as Stat
# using MAT
# using ProgressMeter
using Printf
import HDF5
include("../utilsv2.jl")
include("../streakPH.jl")

# Define constants
const h::Float64 = 4.13566766; # 
const F_AU::Float64 = 5.142206707 * 1e10; # V/m to a.u.
const T_AU::Float64 = 0.024189; # fs to a.u. 
const E_AU::Float64 = 27.2114; # eV to a.u.
const c0::Float64 = 0.29979; # in um/fs

const Ip1::Float64 = 13.6; # Ionization potential

# Functions to build E-field
E_Gauss(t_T0;w=1,tau=1,phi=0) = exp( -4*log(2) * ((t_T0)/tau)^2 ) * cos(w*t_T0 + phi);

function build_Efield(taxis::Vector{T}, T_0::Vector{T0}, Efunc::Function; 
                      kwargs...)::Array{T} where {T0<:Real, T<:Real}
    t_T0 = taxis .- T_0';
    @. Efunc(t_T0; kwargs...)
end

function sample_arrival_time(;lambdaL_um, Theta0_deg)
    w_L = 2*pi*(c0/lambdaL_um); # Streaking Laser frequency, in rad/fs
    T_0 = Theta0_deg .* (pi/(180*w_L));
    return w_L, T_0
end

function prep_ALgaussian(w_L, tauL_fs, taxis)
    A_L = Array{Float64}(undef, length(taxis), 2);
    for i in 1:2
        A_L[:,i] .= view(build_Efield(taxis, [0.0], E_Gauss; w=w_L,
                                      tau=tauL_fs, phi=(i-2)*pi/2), :, 1);
    end
    return A_L
end

function prep_EXgaussian(hvX_eV::Vector{T}, tauX_fs::Real, T0_fs::Vector{T}; 
                         streak::Bool=true, Nt_per_cycle::Int=36, 
                         Twindow=(-3,5), phi::Real=0) where {T<:Real}
""" 
Prepare E-field of Gaussian pulses, using function E_Gauss
Parameters
----------
hvX_eV: central photon energies
tauX_fs: pulse duration, FWHM of E_X(t)
T0_fs: arrival time of the pulses
streak: to streak or not, if not, T_0 will be forced to be [0.]
Nt_per_cycle: number of time points in one optical cycle of the pulse. 
Twindow: in fs, Time window of the simulation. 
         Note this window has to cover all the ionization pulses, but not necessarily the whole streaking pulse,
         because the integrand in the SFA integral is bounded by the |E_X|, as long as |E_X| vanishes (<some threshold), 
         no more contribution will come from later time.
phi: phase (w.r.t streaking field) of the carrier frequency in E_X(t)
Returns
----------
E_X: Array of size (Nt, NT0, Npulse) where Nt=len(taxis), NT0=len(arrival_timeT0), Npulse=[Number of pulses]
     E-field of ionization pulses. Each pulse is scanned at NT0 arrival times
taxis: Vector of size (Nt, ).
     The uniformly sampled time axis, with dt determined by Nt_per_cycle
T_0: Vector of size (NT0, )
     The arrival time of ionization pulses
"""
    hvXc_eV::Real = hvX_eV[div(length(hvX_eV),2)+1];
    dt::Real = (h/hvXc_eV) / Nt_per_cycle; # sampling dt in fs
    T_0 = streak ? T0_fs : [0.0] ;
    taxis = collect(Twindow[1]:dt:Twindow[2]);
    E_X = [];
    for hvX in hvX_eV
        EX = build_Efield(taxis, T_0, E_Gauss; 
                          w=hvX * (2*pi/h), tau=tauX_fs, phi=phi);
        push!(E_X, EX);
    end
    E_X = stack(E_X);
    return E_X, taxis, T_0
end

# Functions for output
function cache_config(h5path, config)
    HDF5.h5open(h5path, "w") do cache_h5
        cg = HDF5.create_group(cache_h5, "config")
        for ky in ["praxis", "qaxis", "Pz_slice"]
            vals = collect(config[ky]);
            # dset = HDF5.create_dataset(cg, ky, Float64, size(vals))
            HDF5.write(cg, ky, vals)
            # println(ky," written")
        end
        HDF5.attributes(cg)["lam_L_um"] = streak_wvl;
        HDF5.attributes(cg)["Ip_eV"] = config["Ip"];
        HDF5.attributes(cg)["Gamma_invfs"] = config["Gamma"];
        for ky in ["tauX", "tauL"]
            if ky in keys(config)
                HDF5.write(cg, ky, config[ky]);
            end
        end
        HDF5.create_group(cache_h5, "densities")
    end
end

function cache_scanvars(h5path; kwargs...)
    HDF5.h5open(h5path, "r+") do cache_h5
        sc = HDF5.create_group(cache_h5, "scan")
        for (ky, val) in Dict(kwargs)
            ky = String(ky);
            # println(ky)
            dset = HDF5.create_dataset(sc, ky, eltype(val), size(val))
            HDF5.write(dset, val)
        end
    end
end
function save_density(h5path, densities, tag::String; attrs...)
    HDF5.h5open(h5path, "r+") do cache_h5
        dg = cache_h5["densities"]
        dset = HDF5.create_dataset(dg, tag, eltype(densities), size(densities))
        HDF5.write(dset, densities)
        for (ky, val) in Dict(attrs)
            ky = String(ky)
            HDF5.attributes(dset)[ky] = val
        end
    end
end



#####Main input parameters######
const STRK::Bool = false; # To simulate the streaked or unstreaked
const PLOT::Bool = false; # To visualize the distributions or not
const hvX_eV_ = [54.4]; #collect(0:-0.2:-2)[1:end-1] .+ 42.8; # central photon energies of the gaussian pulses
const Tw = (-3, 5); # in fs, time window of simulation
const streak_wvl::Float64 = 1.85; # in um, central wavelength of streaking field
const out_h5path::String = get_outdir(@__DIR__) * (STRK ? "/phAnz.h5" : "/phA0p0.h5"); # path to the output h5 file
const tauX::Float64 = 0.3;

# Create pulses
w_L, T_0 = sample_arrival_time(lambdaL_um=streak_wvl, Theta0_deg=collect(-90:2.0:90)[2:end]);

E_X, taxis, T_0 = prep_EXgaussian(hvX_eV_, tauX, T_0, streak=STRK, Twindow=Tw);

A1_L = prep_ALgaussian(w_L, 1e3, taxis);
E_X .*= 1/200; # Global scaling

# Configurate the ROI in momentum space
config = Dict{String, Any}("Gamma"=>0.0)
const dpr::Float64 = STRK ? 5e-3 : 2e-3 
const dpz::Float64 = 0.2 
const Npth::Int = 180; # Most of the time 180 is converged
config["dipole_matrix"] = dipole_M_H; # defined in utilsv2.jl
config["Kmax"] = STRK ? 2.9 : 2.2;  # in a.u.
config["Kmin"] = 1.; # 0.32; # in a.u.
config["Ip"] = Ip1;
config["tauX"] = tauX;

function prep_config_Pgrid(config::Dict{String, Any}; Pz_slice=0)
    Prmax::Float64 = sqrt(2*config["Kmax"]);
    Prmin::Float64 = sqrt(2*get(config,"Kmin",0.));
    Npr::Int = div(Prmax-Prmin, dpr);# 512*2;
    config["dpz"] = dpz;
    praxis, qaxis, Pxym = gen_PxyPolgrid(Prmax, Npr; 
                                         Pmin=Prmin, Nth=Npth);
    config["Np"] = Pxym;
    config["praxis"] = praxis;
    config["qaxis"] = qaxis;
    if isa(Pz_slice, Real)
        # Slice the probability density at a specific Pz
        config["Pz_slice"] = Pz_slice;
    else 
        # The probability density will be integrated over Pz
        Pzmax::Float64 = sqrt(2*(config["Kmax"] - get(config,"Kmin",0.0)));
        config["Pz_slice"] = dpz/2:dpz:Pzmax;
    end
end

prep_config_Pgrid(config, Pz_slice="VMI")
# anything but a real number for Pz_slice results in integration over Pz
# such as Pz_slice="VMI"
cache_config(out_h5path, config)

function scanAmax(Amax::Vector{TA}, E_X::Array{T}, A1_L::Array{T}, taxis::Vector{T}, config::Dict{String,Any}; 
                  out_h5path::String, pulse_kwargs::NamedTuple) where {T<:Real, TA<:Real}
""" Scan ALmax in photoelectron streaking simulations 
Parameters
----------
Amax: Vector of size (Na, )
      Maximal streaking amplitude to be scanned.
E_X:  Array of size (Nt, NT0, Npulse) where Nt=len(taxis), NT0=len(arrival_timeT0), Npulse=[Number of pulses]
      E-field of ionization pulses. Each pulse is scanned at NT0 arrival times
A1_L: Array of size (Nt, 2)
      (x,y) components of the Streaking Vector Potential, in a.u. The maximum must be 1a.u.

Output is written into out_h5path.

Return
-----------
densities: Array of size (NT0, Npulse, Npr, Ntheta)
           The distribution densities of streaked photoelectron, for the (NT0, Npulse) ionizing pulses, 
           at the last streaking amplitude in Amax.
"""
    # Ztype::String = isa(config["Pz_slice"], Real) ? "slice" : "proj";
    local EXgridsize = size(E_X)[2:end];
    E_X_reshaped = reshape(E_X, size(E_X,1), :);
    densities::Array{Float64} = fill(0., 1);

    cache_scanvars(out_h5path; Theta0_deg=T_0.*((180*w_L)/pi), 
                   Amax=Amax, pulse_kwargs...)
    for (i,Amax) in enumerate(Amax_)
        # fname::String = @sprintf("./phA%s%s.mat", f2str(Amax), Ztype);
        A_L = A1_L .* Amax;
        densities = PHInt_Pspace(E_X_reshaped, A_L, taxis, config);
        densities = reshape(densities, EXgridsize..., 
                            length(config["praxis"]), :);
        save_density(out_h5path, densities, "A"*f2str(Amax); ALmax=Amax)
    end
    return densities
end

const Amax_full = vcat(collect(0:0.005:0.015)[2:end],
                       collect(0.02:0.02:0.1),collect(0.12:0.04:0.5));
const Amax_ = STRK ? Amax_full : [0.];
densities = scanAmax(Amax_, E_X, A1_L, taxis, config, 
                     out_h5path=out_h5path, pulse_kwargs=(hvX_eV=hvX_eV_,))
# If the pulse shapes are numbered by some other parameter, just replace hvX_eV with that