using PyCall, SparseArrays, Laplacians, Statistics, LinearAlgebra

function scipyCSC_to_julia(A, B, Size, edges, vgraph)
    #Adjacency matrix
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    AdjMatrix = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)


    laplacian = lap(AdjMatrix)
    Dims = size(laplacian)
    sol = approxchol_lap(AdjMatrix)

    #Incidence matrix
    m, n = B.shape
    colPtr = Int[i+1 for i in PyArray(B."indptr")]
    rowVal = Int[i+1 for i in PyArray(B."indices")]
    nzVal = Vector{Float64}(PyArray(B."data"))
    BB = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)


    k = round(Int64, 6*log2(n)/((0.01)^2/2 - (0.01)^3/3))
    
    Q = Matrix{Float64}(undef, k, Size)
    #Build the random projection matrix
    value = sqrt(3)/sqrt(k)
    for i in 1:k
        for j in 1:Size
            x = rand(1:6)
            if x == 1
                Q[i,j] = -value
            elseif x == 6
                Q[i,j] = value
            else
                Q[i,j] = 0
            end
        end
    end


    Y = Q * BB


    Z = Matrix{Float64}(undef, Dims[1], k)
    for i in 1:k
        yi = Y[i,:]
        zi = sol(yi, tol=1e-12)
        Z[:,i] = zi
    end

    results = Dict{Tuple, Float64}()

    for x in edges
        x1 = convert(Int64, x[1])
        x2 = convert(Int64, x[2])
        
        #+1 in both because indexes start at 1
        toNorm = Z[x1+1,:] - Z[x2+1,:]
        twonorm = norm(toNorm)
        toret = twonorm * twonorm

        results[(vgraph[x1+1], vgraph[x2+1])] = round(toret, digits=2)
    
    end

    return results
end