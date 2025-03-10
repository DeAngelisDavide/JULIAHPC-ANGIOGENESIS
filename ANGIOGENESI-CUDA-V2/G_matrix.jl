using CUDA
function G_kernel!( h::Float64, G )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(G, 1)
    if 1 < i < M
        @inbounds G[i, i-1] = -1 /((2 * h))
        @inbounds G[i, i+1] = +1 /((2 * h))
    end
    return nothing
end