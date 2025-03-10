using CUDA
include("Equations.jl")


function Nnl(U, alpha1, alpha2, L, G, k1, T, k3, k5, k2, nThr, nBlk)
    ncU = N_C(U, alpha1, alpha2, L, G, k1, nThr, nBlk)
    M = size(L, 1)
    npU =  CuArray{Float64}(undef, M)
    niU =  CuArray{Float64}(undef, M)
    nfU = CuArray{Float64}(undef, M)
    CUDA.@sync @cuda threads=nThr blocks=nBlk  N_FIP_kernel!(U, k2, k3, k5, T, nfU, niU, npU)
    nU = CuArray{Float64}(undef, 4*M)
    @cuda threads=nThr blocks=nBlk*4 vcatU!(ncU, npU, niU, nfU, nU)
    return nU
end

function U_update_kernel!(UPrev , tau::Float64, term ,  nU , UNext )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    #4M
    M = size(UNext, 1)
    if i <= M
        @inbounds  UNext[i] = UPrev[i] + tau * (term[i] + nU[i])
    end
    return nothing
end

function U_update!(UPrev, A, tau,  nU, nThr, UNext)
    #A 4M x 4M
    M = size(A, 1)
    term = CuArray{Float64}(undef, M)
    CUBLAS.gemv!('N', 1.0, A, UPrev, 1.0, term)
    #Dimension 4M, increase nBlocks
    @cuda threads=nThr blocks= Int32(ceil( M / nThr)) U_update_kernel!(UPrev, tau, term, nU, UNext)
    return nothing
end

