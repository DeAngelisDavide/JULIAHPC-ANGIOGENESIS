function N_P(U, T, k3, k5)
    M = div(size(U, 1), 4)
    P = U[M+1:2*M]
    I = U[2*M+1:3*M]
    return -k3 .* P .* I .+ k5 .* T
end
