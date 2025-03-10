function L_matrix(h, M)
    L = zeros(M, M)
    L[1, 1] = -2
    L[1, 2] = 2
    for i in 2:M-1
        L[i, i-1] = 1
        L[i, i] = -2
        L[i, i+1] = 1
    end
    L[M, M] = -2
    L[M, M-1] = 2
    return L / h^2
end
