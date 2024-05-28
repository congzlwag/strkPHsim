using LinearAlgebra
import Statistics as Stat
# using MAT
# using ProgressMeter
using Printf
import HDF5
include("../utilsv2.jl")
include("../io_utils.jl")
include("../streakPH2.jl")

# Define constants
const h::Float64 = 4.13566766; # 
const F_AU::Float64 = 5.142206707 * 1e10; # V/m to a.u.
const T_AU::Float64 = 0.024189; # fs to a.u. 
const E_AU::Float64 = 27.2114; # eV to a.u.
const c0::Float64 = 0.29979; # in um/fs

const Ip1::Float64 = 24.6; # Ionization potential
const Ip2::Float64 = 54.4;

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
function cache_config(h5path, config::Dict)
    HDF5.h5open(h5path, "w") do cache_h5
        HDF5.create_group(cache_h5, "densities")
        cg = HDF5.create_group(cache_h5, "config")
        dict2group(cg, config)
        write_h5obj(cg, "lam_L", streak_wvl)
        tag_unit(cg, ["lam_L"], "um");
        tag_unit(cg, ["Ip", "Ip_double"], "eV");
        tag_unit(cg, ["Gamma", "Gamma_double"], "1/fs");
    end
end
function cache_scanvars(h5path; kwargs...)
    HDF5.h5open(h5path, "r+") do cache_h5
        sc = HDF5.create_group(cache_h5, "scan")
        dict2group(sc, Dict(kwargs))
        tag_unit(sc, ["Theta0"], "deg");
        tag_unit(sc, ["hvX"], "eV");
    end
end
function save_density(h5path, densities::Array{T}, tag::String; attrs...) where {T <: Real}
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

# TODO: Recommend Kmin, Kmax for P1, P2
function recommend_Kbounds(hvX_eV, IP1, IPdouble)
    return
end


# Visualization functions
using Plots
function visualize_pulses(E_X)
    tmp = @view E_X[:, div(size(E_X, 2)+1,2), 1:3:end]
    p1 = vis_pulses(tmp)
    title!(p1, "Central wX scan")
    pshow = [p1,]
    if STRK
        tmp = @view E_X[:, 1:10:end, 1]
        p2 = vis_pulses(tmp)
        title!(p2, "T0 scan")
        push!(pshow, p2)
    end
    plt = plot(pshow...)
    gui(plt)
end
function vis_pulses(Efields)
    plot(taxis, Efields, yaxis="EX field [a.u.]")
    pA = plot!(twinx(), taxis, A1_L, yaxis="AL field [a.u.]", xaxis="t [fs]",
               label=["Ax" "Ay"], color=[:black :red])
    p = plot(pA)
    xlims!(-2,2);
    return p
end

function visualize_densities(densities, config)
    (NT0, Npulses) = size(densities)[1:2];
    t0c::Int = div(NT0+1,2);
    hvc::Int = div(Npulses+1,2);
    p1 = vis_density(view(densities, t0c, hvc, :, :), config, t0c, hvc);
    plt = [p1, ]
    if NT0 > 1
        tmp = @view densities[end, hvc, :,:];
        p2 = vis_density(tmp, config, NT0, hvc);
        push!(plt, p2)
    end
    if Npulses > 1
        Npulses = size(densities,2);
        tmp = @view densities[t0c, end, :,:];
        p3 = vis_density(tmp, config, t0c, Npulses)
        push!(plt, p3)
        if NT0 > 1
            tmp = @view densities[end, end, :,:];
            p2 = vis_density(tmp, config, NT0, Npulses);
            push!(plt, p2)
        end
    end
    plt = plot(plt...)
    gui(plt)
end
function vis_density(density, config, idT0::Int, idEX::Int)
    p = heatmap(config["qaxis"], config["praxis"], density, c=:heat)
    title!(p, @sprintf("(Theta0, id_EX)=(%ddeg, %d)", T_0[idT0] * (180/pi*w_L), idEX))
    p
end


#####Main input parameters######
const STRK::Bool = true; # To simulate the streaked or unstreaked
const PLOT::Bool = true; # To visualize the distributions or not
const hvX_eV_ = [Ip2+13.7,]# #collect(-1:1.0:1) .+ 68; # central photon energies of the gaussian pulses
const Tw = (-3, 5); # in fs, time window of simulation
const streak_wvl::Float64 = 1.85; # in um, central wavelength of streaking field
const out_h5path::String = "./phstreak.h5" # path to the output h5 file

const dpr::Float64 = 2e-2
const dKz2::Float64 = 0.15
const Npth::Int = 40; # Most of the time 180 is converged

# Create pulses
w_L, T_0 = sample_arrival_time(lambdaL_um=streak_wvl, Theta0_deg=collect(-90:30:90)[2:end]);

E_X, taxis, T_0 = prep_EXgaussian(hvX_eV_, 0.3, T_0, streak=STRK, Twindow=Tw);

A1_L = prep_ALgaussian(w_L, 1e3, taxis);
E_X .*= 1/200; # Global scaling

# Configurate the ROI in momentum space
config = Dict{String, Any}("Gamma"=>1/3.1) # Linewidth of He+1s doi.org/10.1063/1.5022479
config["dipole_matrix"] = dipole_M_H; # defined in utilsv2.jl
config["P1mesh"] = Dict{String, Any}("dpr" => dpr)
# TODO: Recommend Kmin, Kmax based on hvX_eV_ and Ip1, Ip2
config["P1mesh"]["Kmax"] = 2.3^2/2 # in a.u.
config["P1mesh"]["Kmin"] = 1.5^2/2 # in a.u.
config["P1mesh"]["dKz2"] = dKz2;
config["P1mesh"]["Nth"] = Npth;
config["P1mesh"]["Pz"] = 0;
config["Ip"] = Ip1;
config["Gamma_double"] = 0.0;
config["dipole_matrix_double"] = dipole_M_H; # defined in utilsv2.jl
config["P2mesh"] = Dict{String, Any}("dpr" => dpr)
config["P2mesh"]["Kmax"] = 1.6^2/2 # in a.u.
config["P2mesh"]["Kmin"] = 0.4^2/2 # in a.u.
config["P2mesh"]["dKz2"] = dKz2;
config["P2mesh"]["Nth"] = Npth;
config["P2mesh"]["Pz"] = 0;
config["Ip_double"] = Ip2+Ip1;
# config["accumPz"] = false;


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

    cache_scanvars(out_h5path; Theta0=T_0.*((180*w_L)/pi), 
                   Amax=Amax, pulse_kwargs...)
    for (i,Amax) in enumerate(Amax_)
        # fname::String = @sprintf("./phA%s%s.mat", f2str(Amax), Ztype);
        A_L = A1_L .* Amax;
        densities = PH2Int(E_X_reshaped, A_L, taxis, config);
        densities = reshape(densities, 
                            length(config["P2mesh"]["praxis"]), length(config["P2mesh"]["qaxis"]),
                            EXgridsize..., 
                            length(config["P1mesh"]["praxis"]), length(config["P1mesh"]["qaxis"]));
        if i==1
            HDF5.h5open(out_h5path, "r+") do cache_h5
                write_h5obj(cache_h5, "config", config)
            end
        end
        save_density(out_h5path, densities, "A"*f2str(Amax); ALmax=Amax)
    end
    return densities
end

Amax_ = STRK ? [0.1,] : [0.];
densities = scanAmax(Amax_, E_X, A1_L, taxis, config, 
                     out_h5path=out_h5path, pulse_kwargs=(hvX=hvX_eV_,))
# densities = reshape(densities, :, Npth, size(densities)[2:end]...);
p2marg = dropdims(sum(densities; dims=(5,6)); dims=(5,6));
p2marg = permutedims(p2marg, [3,4,1,2]);
p1marg = dropdims(sum(densities; dims=(1,2)); dims=(1,2));
# visualize_densities(view(densities,1,:,:,:,:), config["P1mesh"])
visualize_densities(p2marg .* 1e8, config["P2mesh"])


# If the pulse shapes are numbered by some other parameter, just replace hvX_eV with that

# if PLOT
#     visualize_pulses(E_X)
#     visualize_densities(densities, config)
# end