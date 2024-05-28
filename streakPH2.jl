using FLoops

function PH2Int(E_X::Array{Float64,2}, A::Array{Float64,2}, 
         t_X::Vector{Float64}, config::Dict{String, Any} )::Array{Float64}
"""
    Calculate complex amplitude of Photoelectron, based on the SFA model
    E_X: ionization field size(E_X) = (Nt, NT0)
    A  : streaking field size(E_X) = (Nt, 2), A[:,1]= Ax, A[:,2]= Ay
    t_X: time axis for both fields size(t_X) = (Nt,)
    config: a dictionary containing the following parameters:
        Ip::Real
        Gamma::Real
        Kmax::Real, already in a.u.
        (Optional, 0) Kmin::Real, already in a.u.
        Np::Int or Array{Real}(undef, 2, Npi)
        (Optional, nothing) Pz_slice:: Real
            if nothing, then eventually a pz-axis will be assigned here
        dipole_matrix::Function
    return PH_p: size(PH_p) = (Npz, NT0, Npi)
"""
    # Apply Configurations and convert units
    IP1::Float64 = config["Ip"] / E_AU; # Change eV to a.u.
    IP2::Float64 = config["Ip_double"] / E_AU; # Change eV to a.u.
    Gamma1::Float64 = config["Gamma"] * T_AU;
    Gamma2::Float64 = config["Gamma_double"] * T_AU;
    dipole01_func::Function = config["dipole_matrix"];
    dipole12_func::Function = config["dipole_matrix_double"];
    # Whether to integrate over Pz or not
    Zaccum::Bool=get(config, "accumPz", true); 
    
    t_X = t_X / T_AU; 
    t_X .-= t_X[1];
    # From now on, everything is in a.u.
    
    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    N_t::Int = size(t_X,1); # Number of time points
    dt = check_taxis(t_X);
    
    # Setup momentum grid for p1 and p2
    P_xm1, P_ym1, P_z1 = parse_Ppolargrid(config["P1mesh"]);
    N_pi1 = length(P_xm1); N_pz1 = length(P_z1);
    P_xm2, P_ym2, P_z2 = parse_Ppolargrid(config["P2mesh"]);
    N_pi2 = length(P_xm2); N_pz2 = length(P_z2);

    # Allocate output shape=(Npz2, Npi2, Npz1, Npi1, NT0)
    PH2density::Array{Float64} = fill(0.0, (Zaccum ? 1 : N_pz2), N_pi2, 
                                      (Zaccum ? 1 : N_pz1), NT0, N_pi1);
    # Allocate temporary vectors to be reused for every Pxy1
    ionEt::Vector{ComplexF64} = fill(0.0, N_t);
    vkvxy_ionEt::Vector{ComplexF64} = fill(0.0, N_t);
    ph_amp::Vector{ComplexF64} = fill(0.0, N_t);
    vxy1::Array{Float64} = fill(0.0, N_t, 2);
    vxy1sq::Array{Float64} = fill(0.0, N_t);
    vkvIxy1::Vector{Float64} = fill(0.0, N_t);
    kzphase1::Vector{Float64} = fill(0.0, N_t);
    diple01::Vector{Float64} = fill(0.0, N_t);
    dionEt::Vector{ComplexF64} = fill(0.0, N_t);
    tmpPH2::AbstractArray{Float64} = fill(0.0, 1);
    if Zaccum # Allocate a separate array
        tmpPH2 = fill(0.0, 1, N_pi2, N_pz1);
    end

    # Calc Volkov phase of P2, to be reused at every P1
    Vxy2::Array{Float64} = fill(0.0, N_t, 2, N_pi2);
    Vxy2sq::Array{Float64} = fill(0.0, N_t, N_pi2);
    VkvIxy2::Array{Float64} = fill(0.0, N_t, N_pi2);
    calcVolkovXYPhase(P_xm2, P_ym2, A, dt; 
                      Vxy=Vxy2, Vxy_sq=Vxy2sq, VkvIxy=VkvIxy2);
    Zphase2::Array{Float64} = @. t_X * ((P_z2' ^ 2) /2.0);
    # Loop the P1xy grid
    for (ind_xy, (Px1, Py1)) in enumerate(zip(P_xm1, P_ym1))
        # vkvIxy1 = xy-Volkov phase 
        # Meanwhile, vxy1sq will be cached with Vx^2+Vy^2
        #            vxy1 will be cached with Vx, Vy
        calcVolkovXYPhase(Px1, Py1, A, dt;
                          Vxy=vxy1, Vxy_sq=vxy1sq, VkvIxy=vkvIxy1);
        vx = @inbounds view(vxy1,:,1);
        vy = @inbounds view(vxy1,:,2);
        for ipulse in 1:NT0
            Ei = view(E_X,:,ipulse); 
            # ionEt = Multiply the cation phasor on the pulse
            # This is a P-independent factor in the inner integral
            mult_ionphase_EX(IP1-1im*Gamma1/2, t_X, Ei, ionEt); 
            # dionEt = Multiply the dication-cation phasor on the pulse
            # This is a P-independent factor in the outer integral
            mult_ionphase_EX((IP2-1im*Gamma2/2) - (IP1-1im*Gamma1/2), t_X, Ei, dionEt);
            # This is the Pz1-independent factor in the inner integrand
            @. vkvxy_ionEt = exp(1im * vkvIxy1) * ionEt;
            if ~Zaccum
                tmpPH2 = @inbounds view(PH2density, :,:,:,ipulse,ind_xy);
            end
            for (ind_z, Pz1) in enumerate(P_z1)
                @. diple01 = dipole01_func(vx, vy, vxy1sq+(Pz1^2); Ip_au=IP1); 
                @. kzphase1 = (Pz1^2/2.0) * t_X;
                PHamp_tdep(vkvxy_ionEt, kzphase1, diple01, dt, ph_amp);
                outslice = @inbounds view(tmpPH2, :,:,ind_z);
                scanP2grid(VkvIxy2, P_z2, dionEt, dipole12_func, IP2-IP1,
                           ph_amp, dt, Zaccum; 
                           Vxy=Vxy2, Vxy_sq=Vxy2sq, zphase=Zphase2, outInt=outslice)
            end
            if Zaccum
                # integrate tmpPH2 and copy to the slice in PH2density
                @inbounds PH2density[:,:,1,ipulse,ind_xy] .= pz_integrate(tmpPH2, P_z1; dim=3);
            end
        end
    end
    if Zaccum
        # Drop the redundant dimensions. This doesn't make a copy
        return dropdims(PH2density; dims=(1,3));
    end
    return PH2density
end

function check_taxis(t_X::Vector{T})::Union{T, Vector{T}} where {T <: Real}
    difftX = diff(t_X);
    # Uniform_Step_Size::Bool = true;
    if Stat.std(difftX)  < 1E-3 * abs(Stat.mean(difftX)) # Check if step size is uniform
        dt = (t_X[end]-t_X[1]) / ( length(t_X) - 1 ); # println("Uniform t grid")
    else # Variable step size
        dt = t_X;
    end
    return dt
end

function parse_Ppolargrid(mesh_cfg::Dict)
    Prmax::Float64 = sqrt(2*mesh_cfg["Kmax"]);
    Prmin::Float64 = sqrt(2*get(mesh_cfg,"Kmin",0.));
    dpr::Float64 = mesh_cfg["dpr"];
    Nth::Int32 = get(mesh_cfg, "Nth", 360);
    dq::Float64 = 2*pi / Nth;
    praxis = Prmin .+ collect(1:div(Prmax-Prmin, dpr)) .* dpr;
    qaxis = collect(0:dq:2*pi-dq/2);
    Pxym = [[pr * f(q) for pr in praxis, q in qaxis]
            for f in [cos, sin]];
    mesh_cfg["praxis"] = praxis;
    mesh_cfg["qaxis"] = qaxis .* (180/pi);
    Pz = get(mesh_cfg, "Pz", nothing);
    if isa(Pz, Vector)
        mesh_cfg["pzaxis"] = Pz;
        return (Pxym[1], Pxym[2], Pz)
    end
    Kz2max::Float64 = 2*(mesh_cfg["Kmax"] - get(mesh_cfg, "Kmin", 0.0));
    if isa(Pz, Real)
        Pz = Vector{Float64}([Pz,]);
    elseif "dKz2" in keys(mesh_cfg)
        Pz = sqrt.(0:mesh_cfg["dKz2"]:Kz2max);
    elseif "dpz" in keys(mesh_cfg)
        Pz = 0:mesh_cfg["dpz"]:sqrt(Kz2max);
    else
        Pz = [0.,]
    end
    mesh_cfg["pzaxis"] = Pz;
    return (Pxym[1], Pxym[2], Pz)
end


function calcVolkovXYPhase(Px::Real, Py::Real, A::Array{T,2}, dt::Union{T, Vector{T}}; 
                           Vxy::AbstractArray{T,2}, Vxy_sq::AbstractArray{T,1}, 
                           VkvIxy::AbstractArray{T,1}) where {T <: Real}
    view(Vxy, :, 1) .= Px .- view(A, :, 1);
    view(Vxy, :, 2) .= Py .- view(A, :, 2);
    Vxy_sq .= sum(Vxy .^ 2, dims=2);
    VkvIxy .= cum_Integrate(Vxy_sq ./ 2, dt)
end

function calcVolkovXYPhase(Px::Real, Py::Real, A::Array{T,2}, 
                           dt::Union{T, Vector{T}})::AbstractArray{T,1} where {T <: Real}
    Vxy = similar(A);
    Vxy_sq = similar(view(Vxy,:,1));
    VkvIxy = similar(view(Vxy,:,1));
    calcVolkovXYPhase(Px, Py, A, dt; Vxy=Vxy, Vxy_sq=Vxy_sq, VkvIxy=VkvIxy)
end

function calcVolkovXYPhase(Pxm::AbstractArray{T}, Pym::AbstractArray{T}, A::Array{T,2}, 
                           dt::Union{T,Vector{T}}; 
                           Vxy::AbstractArray{T,3}, Vxy_sq::AbstractArray{T,2}, 
                           VkvIxy::AbstractArray{T,2}) where {T <: Real}
    for (i, (Px, Py)) in enumerate(zip(Pxm, Pym))
        calcVolkovXYPhase(Px, Py, A, dt; 
                          Vxy=view(Vxy,:,:,i), Vxy_sq=view(Vxy_sq,:,i), 
                          VkvIxy=view(VkvIxy,:,i));
    end
    VkvIxy
end

function mult_ionphase_EX(IP::ComplexF64, taxis::Vector{T}, 
                          Ei::AbstractArray{T}, out_ionEt::AbstractArray{ComplexF64}) where {T <: Real}
    @. out_ionEt = exp(1im * IP * taxis);
    out_ionEt .*= Ei;
end

function PHamp_tdep(sourcexy::Vector{ComplexF64}, zphase::Vector{T}, dipole::Vector{T},
                    dt::Union{T, Vector{T}},
                    outAmp::AbstractArray{ComplexF64}) where {T <: Real}
"""
Time-dependent complex amp of the ion-photoelectron pair
"""
    @. outAmp = exp(1im * zphase) * dipole * sourcexy;
    cum_Integrate!(outAmp, dt, dim=1)
end

function scanP2grid(volkovxy::AbstractArray{T,2}, Pzm::Vector{T}, dionEt::AbstractArray{ComplexF64},
                   dipole_func::Function, IP::Real, source::AbstractArray{ComplexF64},
                   dt::Union{T, Vector{T}}, Zaccum::Bool; 
                   Vxy::AbstractArray{T,3}, Vxy_sq::AbstractArray{T,2}, zphase::Array{T,2},
                   outInt::AbstractArray{T},
                   ) where {T <: Real}
    N_pi2::Int32 = size(volkovxy)[end];
    N_t::Int32 = length(dionEt);
    N_pz::Int32 = size(zphase)[end];
    @floop ThreadedEx() for ind_xy in 1:N_pi2
        @init begin
            diple = Vector{Float64}(undef,N_t);
            density_xy = Vector{Float64}(undef,N_pz);
            intgrand = Vector{ComplexF64}(undef,N_t);
        end
        vkvxy = @inbounds view(volkovxy,:,ind_xy);
        vx = @inbounds view(Vxy,:,1,ind_xy);
        vy = @inbounds view(Vxy,:,2,ind_xy);
        vxysq = @inbounds view(Vxy_sq,:,ind_xy);

        density_xy .= 0;
        for (ind_z, Pz) in enumerate(Pzm)
            zpi = @inbounds view(zphase, :, ind_z);
            @. diple = dipole_func(vx, vy, vxysq+(Pz^2); Ip_au=IP);
            @. intgrand = exp(1im * (vkvxy+zpi)) * diple * dionEt * source;
            increment = abs(only(integrate(intgrand, dt))) ^ 2;
            if Zaccum
                @inbounds density_xy[ind_z] = increment;
            else
                @inbounds outInt[ind_z, ind_xy] = increment;
            end
        end
        if Zaccum
            @inbounds outInt[1, ind_xy] = only(pz_integrate(density_xy, Pzm));
        end
    end
end

function PHInt(E_X::Array{Float64,2}, A::Array{Float64,2}, 
         t_X::Vector{Float64}, config::Dict{String,Any} )::Array{Float64}
"""
Single direct ionization
"""
    # Apply Configurations and convert units
    IP1::Float64 = config["Ip"] / E_AU; # Change eV to a.u.
    Gamma1::Float64 = config["Gamma"] * T_AU;
    dipole01_func::Function = config["dipole_matrix"];
    # Whether to integrate over Pz or not
    Zaccum::Bool=get(config, "accumPz", true); 
    
    t_X = t_X / T_AU; 
    t_X .-= t_X[1];
    # From now on, everything is in a.u.
    
    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    N_t::Int = size(t_X,1); # Number of time points
    dt = check_taxis(t_X);
    
    # Setup momentum grid for p1 and p2
    P_xm1, P_ym1, P_z1 = parse_Ppolargrid(config["P1mesh"]);
    N_pi1 = length(P_xm1); N_pz1 = length(P_z1);

    # Allocate output shape=(Npz2, Npi2, Npz1, Npi1, NT0)
    PHdensity::Array{Float64} = fill(0.0, (Zaccum ? 1 : N_pz1), NT0, N_pi1);
    
    # Loop the P1xy grid
    @floop ThreadedEx() for (ind_xy, (Px1, Py1)) in enumerate(zip(P_xm1, P_ym1))
    # for (ind_xy, (Px1, Py1)) in enumerate(zip(P_xm1, P_ym1))
        @init begin
            ionEt = Vector{ComplexF64}(undef, N_t);
            diple = Vector{Float64}(undef,N_t);
            vxy1 = Array{Float64}(undef, N_t, 2);
            vxy1sq = Array{Float64}(undef, N_t);
            vkvIxy1 = Vector{Float64}(undef, N_t);
            vkvxy_ionEt = Vector{ComplexF64}(undef, N_t);
            intgrand = Vector{ComplexF64}(undef, N_t);
            density_xy = Vector{Float64}(undef, N_pz1);
        end
        # vkvIxy1 = xy-Volkov phase 
        # Meanwhile, vxy1sq will be cached with Vx^2+Vy^2
        #            vxy1 will be cached with Vx, Vy
        calcVolkovXYPhase(Px1, Py1, A, dt;
                          Vxy=vxy1, Vxy_sq=vxy1sq, VkvIxy=vkvIxy1);
        vx = @inbounds view(vxy1,:,1);
        vy = @inbounds view(vxy1,:,2);
        for ipulse in 1:NT0
            Ei = view(E_X,:,ipulse); 
            # ionEt = Multiply the cation phasor on the pulse
            # This is a P-independent factor in the inner integral
            mult_ionphase_EX(IP1-1im*Gamma1/2, t_X, Ei, ionEt); 
            # This is the Pz1-independent factor in the inner integrand
            @. vkvxy_ionEt = exp(1im * vkvIxy1) * ionEt;
            for (ind_z, Pz1) in enumerate(P_z1)
                @. diple = dipole01_func(vx, vy, vxy1sq+(Pz1^2); Ip_au=IP1); 
                @. intgrand = exp(1im * (Pz1^2)/2 * t_X) * diple * vkvxy_ionEt;
                increment = abs(only(integrate(intgrand, dt))) ^ 2;
                if Zaccum
                    @inbounds density_xy[ind_z] = increment;
                else
                    @inbounds PHdensity[ind_z,ipulse,ind_xy] = increment;
                end
            end
            if Zaccum
                @inbounds PHdensity[1,ipulse,ind_xy] = only(pz_integrate(density_xy, P_z1; dim=1));
            end
        end
    end
    if Zaccum
        # Drop the redundant dimensions. This doesn't make a copy
        return dropdims(PHdensity; dims=1);
    end
    return PHdensity
end

function pz_integrate(density::Array{T}, pz_axis::Vector{T1}; dim=1) where {T<:Real, T1<:Real}
    if length(pz_axis) == 1
        return selectdim(density, dim, 1)
    end
    if (pz_axis[1] >= (pz_axis[2]-pz_axis[1])/2)
        selectdim(density, dim, 1) .*= 2; # compensate the 1/2 factor for the first point in trapz
    end
    itrapz(density, pz_axis, dim=dim)
end