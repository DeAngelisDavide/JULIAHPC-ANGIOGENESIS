using LinearAlgebra
using Random
using Plots
using CUDA
using BenchmarkTools

include("Tfun.jl")
include("Phi.jl")
include("G_matrix.jl")
include("L_matrix.jl")
include("A_matrix.jl")
include("Equations.jl")
include("Nnl.jl")

function main()
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
    M, N = 25, 151

    # spatial step size
    h = Lf / (M - 1)
    #spatial step size
    tau = Tf / (N - 1)

    nThr = 5
    nBlk = Int32(ceil(M / nThr))
    #------------------------------------------
    @time begin
    x = CuArray(collect(LinearAlgebra.LinRange(0, Lf, M)))
    #Only for visualization
    t = LinRange(0, Tf, N)

    T = CUDA.zeros(Float64, M)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk T_kernel!(x, epsi0, Lf, T)


    Phix = CUDA.zeros(Float64, M)
    Phixx = CUDA.zeros(Float64, M)
    phi = CUDA.zeros(Float64, M, M)
    @cuda  threads = nThr blocks = nBlk Phix_kernel!(x, T, epsi0, Lf, alpha4, Phix)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk Phixx_kernel!(x, T, epsi0, Lf, alpha4, Phixx)
    @cuda  threads = nThr blocks = nBlk phi_kernel!(Phix, Phixx, h, phi)


    # Mat Phi, G, L
    G = CUDA.zeros(Float64, M, M)
    L = CUDA.zeros(Float64, M, M)
    A = CUDA.zeros(Float64, 4 * M, 4 * M)
    idM = CUDA.zeros(Float64, M, M)
    @cuda  threads = nThr blocks = nBlk G_kernel!(h, G)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk L_kernel!(h, L)
    @cuda  threads = nThr blocks = nBlk * 4 A_kernel!(dC, dP, dI, alpha3, k4, k6, T, L, phi, idM, A)

    # Init  C, P, I, F
    C = CUDA.zeros(Float64, M)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk C_init_kernel!(x, a, C0, C)
    
    P = CUDA.fill(epsi1, M)
    I_ = CUDA.fill(epsi2, M)
    F = CUDA.fill(epsi3, M)

    # forward Euler
    U = CUDA.zeros(Float64, 4 * M, N)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk* 4 vcatU!(C, P, I_, F, @view(U[:, 1]))
    
    for i in 1:(N-1)
        CUDA.@sync nU = Nnl(@view(U[:, i]), alpha1, alpha2, L, G, k1, T, k3, k5, k2, nThr, nBlk)
        CUDA.@sync U_update!(@view(U[:, i]), A, tau, nU, nThr, @view(U[:, i+1]))
    end
end
    U_cpu = Array(U)
    x_cpu = Array(x)

    #  Surface
    figure3 = Plots.surface(t, x_cpu, U_cpu[1:M, :], xlabel=" Time (t)", ylabel=" Space (x)", zlabel="C(x,t)", title=" Surface")
    savefig("superficie_C.png")
end

 main()

