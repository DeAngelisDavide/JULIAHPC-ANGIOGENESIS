using CUDA

#First  derivative of phi
function Phix_kernel!(x , T , epsi0::Float64, L::Float64, alpha4::Float64, Phix )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(x)
        @inbounds Phix[i] =  -epsi0^(-1) * 2 * T[i] * (L - x[i]) / (1 + alpha4 * T[i])
    end
    return nothing
end


function Phixx_kernel!(x , T , epsi0::Float64, L::Float64, alpha4::Float64, Phixx )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(x)
        @inbounds Phixx[i] = -epsi0^(-2) * 2 * T[i] * (2 * (L - x[i])^2 - epsi0 * (1 + alpha4 * T[i])) / (1 + alpha4 * T[i])^2
    end
    return nothing

end


function phi_kernel!(Phix , Phixx , h::Float64, phi )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(Phix, 1)
    if i <= M
        phi[i, i]= Phixx[i]
        if 1 < i < M
            @inbounds phi[i, i-1] = -1 / (2*h) * Phix[i]
            @inbounds phi[i, i+1] = 1 / (2*h) * Phix[i]
        end
    end
    return nothing
end
