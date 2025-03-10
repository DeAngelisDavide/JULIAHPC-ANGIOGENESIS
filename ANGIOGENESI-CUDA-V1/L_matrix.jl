using CUDA

function L_kernel!(h::Float64, L )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(L, 1)
    if i == 1
        @inbounds L[i, i] = -2 / (h^2)
        @inbounds L[i, i+1] = 2 / (h^2)
    elseif i == M
        @inbounds L[i, i] = -2 / (h^2)
        @inbounds L[i, i-1] = 2 / (h^2)
    elseif 1 < i < M
        @inbounds L[i, i-1] = 1 / (h^2)
        @inbounds L[i, i] = -2 / (h^2)
        @inbounds L[i, i+1] = 1 / (h^2)
    end
    return nothing
end
