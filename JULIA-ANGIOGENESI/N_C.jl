function N_C(U, alpha1, alpha2, L, G, k1)
    M = size(U, 1) ÷ 4
    C = U[1:M]
    P = U[M+1:2*M]
    I = U[2*M+1:3*M]
    F = U[3*M+1:4*M]
    xhat = alpha2 .* I .- alpha1 .* F
    return (G * xhat) .* (G * C) .+ (L * xhat) .* C .+ k1 .* (ones(M) .- C) .* C
end
