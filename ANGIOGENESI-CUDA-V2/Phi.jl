using CUDA

function phi_kernel!(x , T , epsi0::Float64, L::Float64, alpha4::Float64, h::Float64, phi )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(phi, 1)
    if i <= M
        phi[i, i]=  -epsi0^(-2) * 2 * T[i] * (2 * (L - x[i])^2 - epsi0 * (1 + alpha4 * T[i])) / (1 + alpha4 * T[i])^2
        if 1 < i < M
            phix = ( -epsi0^(-1) * 2 * T[i] * (L - x[i]) / (1 + alpha4 * T[i]))
            @inbounds phi[i, i-1] = -1 / (2*h) * phix
            @inbounds phi[i, i+1] = 1 / (2*h) * phix
        end
    end
    return nothing
end
