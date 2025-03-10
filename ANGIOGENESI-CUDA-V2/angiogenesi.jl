using LinearAlgebra
using Random
using Plots
using CUDA
using Statistics
include("PTGLA_matrix.jl")
include("Equations.jl")
include("Nnl.jl")

function main()
    #domain limits
    Lf, Tf = 1.0, 1.0

    #diffusion coefficients
    dC, dP, dI = 0.1, 0.1, 0.1

    #cell interaction coefficients
    k1, k2, k3, k4, k5, k6 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    #sensitivity coefficients, scaling coefficient
    alpha1, alpha2, alpha3, alpha4 = 1.0, 1.0, 1.0, 1.0
    epsi0 = 0.1

    #initial conditions
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

    nThrx = 20
    nThry = 20
    nBlockx = Int32(ceil(4M / nThrx))
    nBlocky = Int32(ceil(4M / nThry))
    #------------------------------------------
    CUDA.@time begin
    x = CuArray(collect(LinearAlgebra.LinRange(0, Lf, M)))
    #Only for visualization
    t = LinRange(0, Tf, N)

    #Vec T and Mat Phi, G, L
    T = CuArray{Float64}(undef, M)
    phi = CuArray{Float64}(undef, M, M)
    G = CuArray{Float64}(undef, M, M)
    L = CuArray{Float64}(undef, M, M)
    CUDA.@sync @cuda  threads = (nThr, nThr) blocks = (nBlk, nBlk) PTGL_kernel!(x, epsi0, Lf, h, alpha4, T, L, G, phi)
    A = CuArray{Float64}(undef, 4M, 4M)
    @cuda threads=(nThrx, nThry) blocks=(nBlockx, nBlocky) A_kernel!(dC, dP, dI, alpha3, k4, k6, T, L, phi, A)

    #forward Euler
    U = CuArray{Float64}(undef, 4M, N)
    CUDA.@sync @cuda  threads = nThr blocks = nBlk U_init_kernel!(x, a, C0, epsi1, epsi2, epsi3, @view(U[:, 1]))

    xhat, term1, term2, term3, term4 = CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, 4M)
    for i in 1:(N-1)
        CUDA.@sync @cuda threads=nThr blocks=nBlk  xhat_kernel!(@view(U[:, i]) ,  alpha1, alpha2, xhat)
        CUDA.CUBLAS.gemv!('N', 1.0,  G, xhat, 1.0, term1)
        CUDA.CUBLAS.gemv!('N', 1.0,  G, @view(U[1:M, i]), 1.0, term2)
        CUDA.CUBLAS.gemv!('N', 1.0,  L, xhat, 1.0, term3)
        CUDA.@sync CUDA.CUBLAS.gemv!('N', 1.0,  A, @view(U[:, i]), 1.0, term4)
        CUDA.@sync @cuda threads=nThr blocks=nBlk*4 U_update_kernel!(@view(U[:, i]), term1, term2, term3, term4, k1, k2, k3, k5, tau, G, T, @view(U[:, i+1]))
    end

    #CUDA.@sync @cuda  threads = M blocks = 4 shmem = sizeof(Float64)*M U_update_kernel!(U, x, a, C0, epsi1, epsi2, epsi3, alpha1, alpha2, k1, k2, k3, k5, tau, A, G, L, T, N)
end
    U_cpu = Array(U)
    x_cpu = Array(x)

    #Surface
    figure3 = Plots.surface(t, x_cpu, U_cpu[1:M, :], xlabel=" Time (t)", ylabel=" Space (x)", zlabel="C(x,t)", title=" Surface")
    savefig("superficie_C.png")
end


main()


#=
    @cuda  threads = nThr blocks = nBlk U_init_kernel!(x, a, C0, epsi1, epsi2, epsi3, @view(U[:, 1]))

    #nU = CuArray{Float64}(undef, 4*M)
    xhat, term1, term2, term3, term4 = CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, M), CuArray{Float64}(undef, 4M) 
    #forward Euler
    for i in 1:(N-1)
        CUDA.@sync @cuda threads=nThr blocks=nBlk xhat_kernel!(@view(U[:, i]) ,  alpha1, alpha2, xhat )
        CUDA.CUBLAS.gemv!('N', 1.0,  G, xhat, 1.0, term1)
        CUDA.CUBLAS.gemv!('N', 1.0,  G, @view(U[1:M, i]), 1.0, term2)
        CUDA.CUBLAS.gemv!('N', 1.0,  L, xhat, 1.0, term3)
        CUDA.@sync CUDA.CUBLAS.gemv!('N', 1.0,  A, @view(U[:, i]), 1.0, term4)
        CUDA.@sync @cuda threads=nThr blocks=nBlk*4 U_update_kernel!(@view(U[:, i]), term1, term2, term3, term4, k1, k2, k3, k5, tau, G, T, @view(U[:, i+1]))
    end=#