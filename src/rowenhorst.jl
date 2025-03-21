rowenhorst_grid(ρ, σ, n) =
    range(-sqrt(n-1)*sqrt(σ^2/(1-ρ^2)), sqrt(n-1)*sqrt(σ^2/(1-ρ^2)), n)

function rowenhorst_matrix(ρ, σ, n)
    P, P′ = (Matrix{typeof(ρ)}(undef, n, n) for i in 1:2)
    q = 0.5*(1+ρ)
    ψ = sqrt(n-1) * sqrt(σ^2/(1-ρ^2))
    iterations = n - 2
    P[1,1] = q
    P[1,2] = 1-q
    P[2,1] = 1-q
    P[2,2] = q
    for iter in 1:iterations
        n = iter+1
        @inbounds P′[1,1] = q * (P[1,1]) 
        for s in 2:n
            @inbounds P′[s,1] = q*P[s,1] + (1-q)*P[s-1,1]
        end
        @inbounds P′[n+1,1] = (1-q) * (P[n,1])
        for j in 2:n
            @inbounds P′[1,j] = 0.5*(q*P[1,j]+(1-q)*P[1,j-1])
            for s in 2:n
                @inbounds P′[s,j] = 0.5*(q*P[s,j]+(1-q)*P[s-1,j]+q*P[s-1,j-1]+(1-q)*P[s,j-1])
            end
            @inbounds P′[n+1,j] = 0.5*((1-q)*P[n,j]+q*P[n,j-1])
        end
        @inbounds P′[1,n+1] = (1-q)*P[1,n]
        for s in 2:n
            @inbounds P′[s,n+1] = q*P[s-1,n]+(1-q)*P[s,n]
        end
        @inbounds P′[n+1,n+1] = q*P[n,n]
        for j in 1:n+1
            for i in 1:n+1
                @inbounds P[i,j] = P′[i,j]
            end
        end
    end
    collect(transpose(P))
end
