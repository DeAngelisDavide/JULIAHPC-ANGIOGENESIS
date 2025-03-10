function phi_matrix(Phix, Phixx, h)
    M = size(Phix, 1)
    phi = diagm(Phixx)
    for i in 2:M-1
        phi[i, i-1] = -1 / (2*h) * Phix[i]
        phi[i, i+1] = 1 / (2*h) * Phix[i]
    end
    return phi
end
