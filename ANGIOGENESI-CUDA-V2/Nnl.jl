using CUDA


function U_update_kernel!(UPrev,term1, term2, term3, term4, k1::Float64, k2::Float64, k3::Float64, k5::Float64, tau::Float64, G, T, UNext)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(G, 1)
  
    if i <= M
            @inbounds nU = term1[i] * term2[i] + term3[i] * UPrev[i] + k1 * (1.0 - UPrev[i]) * UPrev[i]
            @inbounds UNext[i] = UPrev[i] + tau * (term4[i] + nU)
            @inbounds  term1[i], term2[i], term3[i], term4[i] = 0, 0, 0, 0
    elseif M + 1 <= i <= 2M
            @inbounds nU = -k3 * UPrev[i] * UPrev[i+M] + k5 * T[i-M]
            #@inbounds nU[i + 2M] = -k3 * U[i + M] * U[i + 2 * M] 
            #@inbounds nU[i + 3M] = -k2 * U[i + M] * U[i + 2 * M]
            #@inbounds  UNext[i] = U[i] + tau * (term4 + nU[i])
            @inbounds UNext[i] = UPrev[i] + tau * (term4[i] + nU)
            @inbounds  term4[i] = 0
    elseif 2M + 1 <= i <= 3M
            @inbounds nU = -k3 * UPrev[i-M] * UPrev[i]
            @inbounds UNext[i] = UPrev[i] + tau * (term4[i] + nU)
            @inbounds  term4[i] = 0
    elseif 3M + 1 <= i <= 4M
            @inbounds nU = -k2 * UPrev[i-2M] * UPrev[i-M]
            @inbounds UNext[i] = UPrev[i] + tau * (term4[i] + nU)
            @inbounds term4[i] = 0
    end
    return nothing
end


function U_update_kernel!(U, x, a::Float64, C0::Float64, epsi1::Float64, epsi2::Float64, epsi3::Float64, alpha1::Float64, alpha2::Float64, k1::Float64, k2::Float64, k3::Float64, k5::Float64, tau::Float64, A, G, L, T, N)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(G, 1)
  
    if i <= M
        #Init U
        if x[i] < a
            @inbounds U[i, 1] = C0
        end
        xhat = CUDA.@cuDynamicSharedMem(Float64, M)
        for jU = 1:(N-1)
            term1, term2, term3, term4 = 0, 0, 0, 0
            xhat[i] = alpha2 * U[i+2*M, jU] - alpha1 * U[i+3*M, jU]
            CUDA.sync_threads()
            for j = 1:M
                @inbounds begin
                    term1 += G[i, j] * xhat[j]
                    term2 += G[i, j] * U[j]
                    term3 += L[i, j] * xhat[j, jU]
                    term4 += A[i, j] * U[j, jU]
                end
            end
            for j = M+1:4M
                @inbounds term4 += A[i, j] * U[j, jU]
            end
            @inbounds nU = term1 * term2 + term3 * U[i, jU] + k1 * (1.0 - U[i, jU]) * U[i, jU]
            @inbounds U[i, jU+1] = U[i, jU] + tau * (term4 + nU)
        end

    elseif M + 1 <= i <= 2M
        @inbounds U[i, 1] = epsi1
        for jU = 1:(N-1)
            @inbounds nU = -k3 * U[i, jU] * U[i+M, jU] + k5 * T[i-M]
            #@inbounds nU[i + 2M] = -k3 * U[i + M] * U[i + 2 * M] 
            #@inbounds nU[i + 3M] = -k2 * U[i + M] * U[i + 2 * M]
            #@inbounds  UNext[i] = U[i] + tau * (term4 + nU[i])
            term4 = 0 
            for j = 1:4M
                @inbounds term4 += A[i, j] * U[j, jU]
            end
            @inbounds U[i, jU+1] = U[i, jU] + tau * (term4 + nU)
        end

    elseif 2M + 1 <= i <= 3M
        @inbounds U[i, 1] = epsi2
        for jU = 1:(N-1)
            @inbounds nU = -k3 * U[i-M, jU] * U[i, jU]
            term4 = 0
            for j = 1:4M
                @inbounds term4 += A[i, j] * U[j, jU]
            end
            @inbounds U[i, jU+1] = U[i, jU] + tau * (term4 + nU)
        end

    elseif 3M + 1 <= i <= 4M
        @inbounds U[i, 1] = epsi3
        for jU = 1:(N-1)
            @inbounds nU = -k2 * U[i-2M, jU] * U[i-M, jU]
            term4 = 0
            for j = 1:4M
                @inbounds term4 += A[i, j] * U[j, jU]
            end
            @inbounds U[i, jU+1] = U[i, jU] + tau * (term4 + nU)
        end
    end
    return nothing
end
