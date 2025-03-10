using CUDA

function U_init_kernel!(x, a::Float64, C0::Float64, epsi1::Float64, epsi2::Float64, epsi3::Float64, U)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = length(x)
    if i <= M
        if x[i] < a
            @inbounds U[i] = C0
        end
        @inbounds U[i+M] = epsi1
        @inbounds U[i+2*M] = epsi2
        @inbounds U[i+3*M] = epsi3
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


function xgemv!(U ,  alpha1::Float64, alpha2::Float64, G, L, A, term1, term2, term3, term4)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(G, 1)
    if i <= M
        lterm1, lterm2, lterm3, lterm4 = 0, 0, 0, 0
        for j=1:M
            xhat = alpha2 *  U[j + 2 * M] - alpha1 * U[j + 3 * M]
            lterm1 += G[i, j] * xhat
            lterm2 = G[i, j] * U[j]
            lterm3 = L[i, j] * xhat
            lterm4 = A[i, j] * U[j]
        end
        for j=M+1:4M
            lterm4 = A[i, j] * U[j]
        end
        term1[i], term2[i], term3[i], term4[i] = lterm1, lterm2, lterm3, lterm4   
    elseif  i <= 2M
        lterm4 = 0
        for j=1:4M
            lterm4 = A[i, j] * U[j]
        end
        term4[i] = lterm4
    elseif  i <= 3M
        lterm4 = 0
        for j=1:4M
            lterm4 = A[i, j] * U[j]
        end
        term4[i] = lterm4
    elseif  i <= 4M
        lterm4 = 0
        for j=1:4M
            lterm4 = A[i, j] * U[j]
        end
        term4[i] = lterm4
    end
end