using LinearAlgebra
using Random
using Plots

include("Tfun.jl")
include("Phixfun.jl")
include("Phixxfun.jl")
include("G_matrix.jl")
include("L_matrix.jl")
include("phi_matrix.jl")
include("A_matrix.jl")
include("Nnl.jl")

function main()
    #domain limits
    Lf, Tf = 1.0, 1.0

    # diffusion coefficients
    dC, dP, dI = 0.1, 0.1, 0.1

    # cell interaction coefficients
    k1, k2, k3, k4, k5, k6 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    # sensitivity coefficients, scaling coefficient
    alpha1, alpha2, alpha3, alpha4 = 1.0, 1.0, 1.0, 1.0
    epsi0 = 0.1

    # initial conditions
    a, C0 = 0.1, 1.0
    epsi1, epsi2, epsi3 = 0.1 * rand(), 0.1 * rand(), 0.1 * rand()

    #number of lines, number of tome steps
    M, N = 250, 1510

    # spatial step size
    h = Lf / (M - 1)
    #spatial step size
    tau = Tf / (N - 1)

    #---------------------------------------------------------
    @time begin
    x = LinRange(0, Lf, M)
    t = LinRange(0, Tf, N)

    T = T_fun(x, epsi0, Lf)
    Phix = Phixfun(x, T, epsi0, Lf, alpha4)
    Phixx = Phixxfun(x, T, epsi0, Lf, alpha4)

    # Mat Phi, G, L
    G = G_matrix(h, M)
    L = L_matrix(h, M)
    phi = phi_matrix(Phix, Phixx, h)
    A = A_matrix(dC, dP, dI, alpha3, k4, k6, T, G, L, phi)


    # Init  C, P, I, F
    C = zeros(M)
    for i in 1:M
        if x[i] < a
            C[i] = C0
        end
    end
    P = fill(epsi1, M)
    I_ = fill(epsi2, M)
    F = fill(epsi3, M)

    # forward Euler
    U = zeros(4 * M, N)
    U[:, 1] .= vcat(C, P, I_, F)

    for i in 1:(N-1)
        nU = Nnl(U[:, i], alpha1, alpha2, L, G, k1, T, k3, k5, k2)
        U[:, i+1] .= U[:, i] + tau * (A * U[:, i] + nU)
    end
end
    #  Surface
    figure3 = Plots.surface(t, x, U[1:M, :], xlabel=" Time (t)", ylabel=" Space (x)", zlabel="C(x,t)", title=" Surface")
    Plots.savefig("fig.png")
    
end

 main()