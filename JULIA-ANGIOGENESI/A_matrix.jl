using LinearAlgebra
function A_matrix(dC, dP, dI, alpha3, k4, k6, T, G, L, phi)
    M = size(T, 1)
    A11 = dC * L - alpha3 * phi
    A21 = k4 * diagm(T)
    #LinearAlgebra.I defined with one byte (Identity Matrix)
    A22 = dP * L - k6 * I
    A33 = dI * L
    Z = zeros(M, M)
    A = [A11 Z Z Z; A21 A22 Z Z; Z Z A33 Z; Z Z Z Z]
    return A
end