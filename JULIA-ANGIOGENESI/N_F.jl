function N_F(U, k2)
    M = div(size(U, 1), 4)
    P = U[M+1:2*M]
    I = U[2*M+1:3*M]
    return -k2 .* P .* I
end
