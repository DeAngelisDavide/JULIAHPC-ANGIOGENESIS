include("N_C.jl")
include("N_P.jl")
include("N_I.jl")
include("N_F.jl")
function Nnl(U, alpha1, alpha2, L, G, k1, T, k3, k5, k2)
    ncU = N_C(U, alpha1, alpha2, L, G, k1)
    npU = N_P(U, T, k3, k5)
    niU = N_I(U, k3)
    nfU = N_F(U, k2)
    return vcat(ncU, npU, niU, nfU)
end
