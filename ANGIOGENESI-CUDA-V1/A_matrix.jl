using CUDA



function A_kernel!(dC::Float64, dP::Float64, dI::Float64, alpha3::Float64, k4::Float64, k6::Float64, T, L, phi, idM, A)
    #A = [A11 Z Z Z; A21 A22 Z Z; Z Z A33 Z; Z Z Z Z]
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M = size(T, 1)

    #A11
    if i <= M
        @inbounds begin
            for j = 1:M
                A[i, j]= dC * L[i, j]  - alpha3 * phi[i, j] 
            end
        end
    #A21, A22
    elseif M+1 <= i <= 2*M
        @inbounds begin
            A[i, i-M] = T[i-M] * k4
            idM[i-M, i-M] = 1
            for j = M+1:2*M
                A[i, j] = dP * L[i-M, j-M] - k6 * idM[i-M, j-M]
            end
        end
    #A33
    elseif 2*M+1 <= i <= 3*M
        ir = i-2M
        @inbounds begin
            for j = 2*M+1:3*M
                A[i, j] = dI * L[ir, j-2M]
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
    idM = @cuDynamicSharedMem(Float64, blockDim().x*M)
    #A11
    if i <= M
        @inbounds begin
            for j = 1:M
                A[i, j]= dC * L[i, j]  - alpha3 * phi[i, j] 
            end
        end
    #A21, A22
    elseif M+1 <= i <= 2*M
        ir = i-M
        @inbounds begin
            A[i, ir] = T[ir] * k4
            for j = M+1:2M
                idM[(threadIdx().x -1) * M + (j-M)] = (i == j) ? 1 : 0
                A[i, j] = dP * L[ir, j-M] - k6 * idM[(threadIdx().x -1) * M + (j-M)]
            end
        end
    #A33
    elseif 2M+1 <= i <= 3M
        ir = i-2M
        @inbounds begin
            for j = 2M+1:3M
                A[i, j] = dI * L[ir, j-2M]
            end
        end
    end
    return nothing
end
=#