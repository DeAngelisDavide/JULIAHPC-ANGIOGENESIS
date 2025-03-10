#=Contraction of Tumor Angiogenic Factors (TAFs)
at poition x with scaling eps
=#
function T_fun(x, eps, L)
    return exp.(-eps^(-1) * (L .- x).^2)
end
