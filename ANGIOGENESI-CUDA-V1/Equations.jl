using CUDA

function  C_init_kernel!(x , a::Float64, C0::Float64, C )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    M = length(x)
    if i <= M
        if x[i] < a
            @inbounds C[i] = C0
        end
    end
    return nothing
end

function vcatU!(C , P , I_ , F , U )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x 
    M = length(C)
    if i <= M
        @inbounds U[i] = C[i]
        @inbounds U[i + M] = P[i]
        @inbounds U[i + 2*M] = I_[i]
        @inbounds U[i + 3*M] = F[i]
    end
    return nothing
end

function xhat_kernel!(U ,  alpha1::Float64, alpha2::Float64, xhat )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = length(xhat)
    if i <= M
        #I[i] = U[i + 2 * M]
        #F[i] =  U[i + 3 * M]
        @inbounds xhat[i] = alpha2 *  U[i + 2 * M] - alpha1 * U[i + 3 * M]
    end
    return nothing
end

function N_C_kernel!(U , term1 , term2 , term3 , k1::Float64, ncU )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = length(ncU)
    if i <= M
        #C[i] = U[i]
        @inbounds ncU[i] = term1[i] * term2[i] + term3[i] * U[i] + k1 * (1.0f0 - U[i]) * U[i]
    end
    return nothing
end


function N_C(U , alpha1::Float64, alpha2::Float64, L , G , k1::Float64, nThreads::Int64, nBlocks::Int32)
    M = size(G, 1)
    xhat =  CuArray{Float64}(undef, M)
    @cuda threads=nThreads blocks=nBlocks xhat_kernel!(U, alpha1, alpha2, xhat)
    term1 = CuArray{Float64}(undef, M) #G * xhat
    term2 = CuArray{Float64}(undef, M) #G * C
    term3 = CuArray{Float64}(undef, M) #(L * xhat)  
    # Moltiplicazione matrice-vettore G * xhat usando cuBLAS
    CUDA.CUBLAS.gemv!('N', 1.0, G, xhat, 1.0, term1)
    # Moltiplicazione matrice-vettore G * C usando cuBLAS
    #C = U[1:M]
    CUDA.CUBLAS.gemv!('N', 1.0, G, @view(U[1:M]),  1.0, term2)
    # Moltiplicazione matrice-vettore L * xhat usando cuBLAS
    CUDA.CUBLAS.gemv!('N', 1.0, L, xhat, 1.0, term3)
    
    ncU = CuArray{Float64}(undef, M)
    @cuda threads=nThreads blocks=nBlocks N_C_kernel!(U, term1, term2, term3, k1, ncU)
    return ncU
end

function N_FIP_kernel!(U , k2::Float64, k3::Float64,  k5::Float64, T , nfU ,  niU , npU )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = div(size(U, 1), 4)
    if i <= M
        #P[i] = U[i + M]    
        #I[i] = U[i + 2 * M]   
        @inbounds nfU[i] = -k2 * U[i + M] * U[i + 2 * M]
        @inbounds niU[i] = -k3 * U[i + M] * U[i + 2 * M]
        @inbounds term1 = -k3 * U[i + M] * U[i + 2 * M] 
        @inbounds term2 = k5 * T[i]                      
        @inbounds npU[i] = term1 + term2        
    end
    return nothing
end

