function G_matrix(h, M)
G = zeros(M, M)
for i in 2:M-1
    G[i, i-1] = -1
    G[i, i+1] = +1
end
return G / (2 * h)
end
