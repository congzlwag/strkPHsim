using FLoops
function PHInt_Pspace(E_X::Array{Float64,2}, A::Array{Float64,2}, 
         t_X::Array{Float64,1}, config::Dict{String, Any} )::Array{Float64}
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
    I_p::Float64 = config["Ip"] / E_AU; # Change eV to a.u.
    Gamma::Float64 = config["Gamma"] * T_AU;
    K_max::Float64 = config["Kmax"];
    K_min::Float64 = get(config, "Kmin", 0.0);
    dipoleM::Function = config["dipole_matrix"];
    Zaccum::Bool=get(config, "accumPz", true);
    
    t_X = t_X / T_AU; 
    t_X .-= t_X[1];
    # From now on, everything is in a.u.
    
    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    N_t::Int = size(t_X,1); # Number of time points
    difftX = diff(t_X);
    # Uniform_Step_Size::Bool = true;
    if Stat.std(difftX)  < 1E-3 * abs(Stat.mean(difftX)) # Check if step size is uniform
        dt = (t_X[end]-t_X[1]) / ( N_t - 1 ); # println("Uniform t grid")
    else # Variable step size
        dt = t_X; # println("Non-uniform t grid")
    end
    difftX = nothing;
    
    # Setup momentum grid
    dpz = config["dpz"];
    @assert haskey(config, "Pz_slice")
    Pz_slice = config["Pz_slice"];
    P_xm, P_ym, P_z = get_Pgrid(config["Np"], Pz_slice, dpz);
    P_xm = collect(P_xm); P_ym = collect(P_ym);
    N_pz = length(P_z); # number of z-points
    N_pi = length(P_xm);
    # if !(haskey(config, "Pz_slice"))
    #     config["Pz_slice"] = P_z;
    # end

    # Allocate memory for PH_p
    hasZaxis::Bool = ~isa(config["Pz_slice"], Real) & ~Zaccum; # Whether the result has Pzaxis
    PH_p::Array{Float64} = hasZaxis ?  fill(0.0, N_pz, NT0, N_pi) : fill(0.0, NT0, N_pi);
    
    #Phase from photoionization
    Phase_PI::Array{ComplexF64,1} = exp.( (1im * I_p + Gamma/2 ).* (t_X .- t_X[1]) ); # size=(N_t,)
    PI_ints::Array{ComplexF64,2} = E_X .* Phase_PI;
    
    P_xm = vec(collect(P_xm)); P_ym = vec(collect(P_ym));
    let dt=dt
        @floop ThreadedEx() for (ind_p, Px,Py) in zip(1:N_pi, P_xm, P_ym)
        # for (ind_p, Px,Py) in zip(1:N_pi, P_xm, P_ym)
            @init begin
                I_xy = Vector{Float64}(undef,N_t);
                diple = Vector{Float64}(undef,N_t);
                V_x = Vector{Float64}(undef,N_t);
                V_y = Vector{Float64}(undef,N_t);
                density_xy = Vector{Float64}(undef,N_pz);
            end
            V_x .= Px .- view(A, :, 1);
            V_y .= Py .- view(A, :, 2);

            @. I_xy = (1/2) * ( V_x^2 + V_y^2 );
            cum_Integrate!(I_xy, dt, dim=1);
            # Volkov phase, w/o changing I_xy because it'll be reused in calculating the dipole
            for ind_calc in 1:NT0
                PI_int = @inbounds @view PI_ints[:,ind_calc]; # size=(N_t,)
                density_xy .= 0;
                for ind_z in 1:N_pz
                    K_z = (1/2) * (P_z[ind_z])^2;
                    if (K_min > (1/2) * (Px^2 + Py^2) + K_z)
                        continue
                    end
                    @. diple = dipoleM(V_x, V_x^2+V_y^2+2*K_z; Ip_au=I_p); 
                    intgrand = (exp(1im * (vlkv+K_z*t)) * Pint*d for (t,vlkv,Pint,d) in zip(t_X,I_xy,PI_int,diple));
                    increment::Float64 = abs(only(sum(intgrand)*dt)) ^2;
                    if hasZaxis
                        PH_p[ind_z,ind_calc,ind_p] = increment;
                    elseif (N_pz==1)
                        PH_p[ind_calc,ind_p] = increment;
                    else 
                        density_xy[ind_z] = increment;
                    end
                end
                if ~(hasZaxis | N_pz==1)
                    PH_p[ind_calc,ind_p] += only(pz_integrate(density_xy, P_z));
                end
            end
        end
    end
    PH_p
end

function pz_integrate(density::Array{Float64}, pz_axis::Union{Vector{T1},StepRangeLen{T1}}) where {T1<:Real}
    if (pz_axis[1] >= (pz_axis[2]-pz_axis[1])/2)
        selectdim(density,1,1) .*= 2; # compensate the 1/2 factor for the first point in trapz
    end
    itrapz(density, pz_axis, dim=1)
end
