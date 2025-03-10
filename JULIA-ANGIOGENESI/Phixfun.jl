#First  derivative of phi
function Phixfun(x, T, epsi0, L, alpha4)
    return -epsi0^(-1) .* 2 .* T .* (L .- x) ./ (1 .+ alpha4 .* T)
end

