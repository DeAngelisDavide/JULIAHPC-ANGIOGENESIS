using CUDA 

#=Contraction of Tumor Angiogenic Factors (TAFs)
at poition x with scaling eps
=#
function T_kernel!(x , eps, L, T )
    # in Julia, threads and blocks begin numbering with 1 instead of 0
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(x)
        @inbounds T[i] =  exp(-eps^(-1) * (L - x[i])^2)
    end
    return nothing
end
