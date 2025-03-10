using CUDA


function PTGL_kernel!(x, epsi0::Float64, Lf::Float64, h::Float64, alpha4::Float64, T, L, G, phi)
    j = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    i = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    M = size(T, 1)
    if i <= M && j <= M
        @inbounds begin
            #Local calculation to avoid syncronization and allowing the return of T
            Ti = exp(-epsi0^(-1) * (Lf - x[i])^2)
            #A11 , L, G, phi, T
            if j == 1
                T[i] = Ti
            end
            if (j == i + 1 || j == i - 1)
                #phi[i, i-1] = -1 / (2*h) * phix | phi[i, i+1] = 1 / (2*h) * phix
                phix = (-epsi0^(-1) * 2 * T[i] * (Lf - x[i]) / (1 + alpha4 * Ti))
                phi[i, j] = 1 * (j - i) / (2 * h) * phix
                L[i, j] = (1 + ((i == 1 && j == 2) || (i == M && j == M - 1) ? 1 : 0)) / (h^2)
                #G[i, i-1] = -1 /((2 * h)) | G[i, i+1] = +1 /((2 * h))  1 < i < M
                G[i, j] = (1 * (j - i) / ((2 * h))) * ((i == 1 || i == M) ? 0 : 1)
            elseif i == j
                ## phi[i, i] = Phixx[i] 
                phi[i, j] = -epsi0^(-2) * 2 * T[i] * (2 * (Lf - x[i])^2 - epsi0 * (1 + alpha4 * Ti)) / (1 + alpha4 * Ti)^2
                L[i, j] = -2 / (h^2)
                G[i, j] = 0
            else
                phi[i, j] = 0
                L[i, j] = 0
                G[i, j] = 0
            end
        end
    end
    return nothing
end


function A_kernel!(dC::Float64, dP::Float64, dI::Float64, alpha3::Float64, k4::Float64, k6::Float64, T, L, phi, A)
    # Calcolo degli indici bidimensionali
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    M = size(T, 1)

    
    if i <= 4M && j <= 4M
        @inbounds begin
            if i <= M
                #[A11, Z, Z, Z]
                if j <= M
                    A[i, j] = dC * L[i, j] - alpha3 * phi[i, j]
                else
                    A[i, j] = 0
                end
            elseif M+1 <= i <= 2M
                #[A21, A22, Z, Z]
                ir = i - M
                if j <= M
                    A[i, j] = (j == ir) ? T[ir] * k4 : 0
                elseif M+1 <= j <= 2M
                    idM = (i == j) ? 1 : 0
                    A[i, j] = dP * L[ir, j - M] - k6 * idM
                else
                    A[i, j] = 0
                end
            elseif 2M+1 <= i <= 3M
                #[Z, Z, A33, Z]
                ir = i - 2M
                if 2M+1 <= j <= 3M
                    A[i, j] = dI * L[ir, j - 2M]
                else
                    A[i, j] = 0
                end
            else
                #[Z, Z, Z, Z]
                A[i, j] = 0
            end
        end
    end

    return nothing
end

#=
function A_kernel!(dC::Float64, dP::Float64, dI::Float64, alpha3::Float64, k4::Float64, k6::Float64, T, L, phi, A)
    #A = [A11 Z Z Z; A21 A22 Z Z; Z Z A33 Z; Z Z Z Z]
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(T, 1)
    #[A11, Z, Z, Z]
    if i <= M
        @inbounds begin
            #A11
            for j = 1:M
                A[i, j] = dC * L[i, j]  - alpha3 * phi[i, j] 
            end
            # Z, Z, Z]
            for j = M+1:4M
                A[i, j] = 0
            end
        end
    #[A21, A22, Z, Z]
    elseif M+1 <= i <= 2*M
        ir = i-M
        @inbounds begin
            #A21
            A[i, ir] = T[ir] * k4
            #A22
            for j = M+1:2M
                idM = (i == j) ? 1 : 0
                A[i, j] = dP * L[ir, j-M] - k6 * idM
            end

            #Z, Z]
            for j = 2M+1:4M
                A[i, j] = 0
            end
        end
    #[Z, Z, A33, Z]
    elseif 2M+1 <= i <= 3M
        ir = i-2M
        @inbounds begin
            #[Z, Z
            for j = 1:2M
                A[i, j] = 0
            end
            #A33
            for j = 2M+1:3M
                A[i, j] = dI * L[ir, j-2M]
            end
            #Z]
            for j = 3M+1:4M
                A[i, j] = 0
            end
        end
    #[Z, Z, Z, Z]
    elseif 3M+1 <= i <= 4M
        @inbounds begin
            for j = 1:4M
                A[i, j] = 0
            end
        end
    end
    return nothing
end
=#

#=
function phi_kernel!(x , T , epsi0::Float64, L::Float64, alpha4::Float64, h::Float64, phi )
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(phi, 1)
    if i <= M
        phi[i, i]=  -epsi0^(-2) * 2 * T[i] * (2 * (L - x[i])^2 - epsi0 * (1 + alpha4 * T[i])) / (1 + alpha4 * T[i])^2
        if 1 < i < M
            phix = ( -epsi0^(-1) * 2 * T[i] * (L - x[i]) / (1 + alpha4 * T[i]))
            @inbounds phi[i, i-1] = -1 / (2*h) * phix
            @inbounds phi[i, i+1] = 1 / (2*h) * phix
        end
    end
    return nothing
end
=#