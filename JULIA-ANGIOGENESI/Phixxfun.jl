#Second derivative of phi
function Phixxfun(x, T, epsi0, L, alpha4)
    return -epsi0^(-2) * 2 .* T .* (2 .* (L .- x).^2 .- epsi0 .* (1 .+ alpha4 .* T)) ./ (1 .+ alpha4 .* T).^2
end
