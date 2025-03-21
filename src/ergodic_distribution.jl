@kwdef @concrete struct ErgodicDistributionProblem
    weights
    solver
    maxiters = 10^4
    abstol = 1e-20
end
@kwdef @concrete struct ErgodicDistributionCache
    prob
    grid
    policies
    distribution = zeros(length.(grid))
    tmp = copy(distribution)
    inds0 = Index.(length.(grid))
    inds1 = ntuple(length(grid)) do i
        i in keys(prob.weights) ? Index(length(grid[i])) : inds0[i]
    end
    weights = Tuple(
        itensor(prob.weights[k], inds1[k], inds0[k])
        for k in keys(prob.weights)
    )
end
init(prob::ErgodicDistributionProblem, grid, policies) =
    ErgodicDistributionCache(; prob, grid, policies)

function solve!(cache::ErgodicDistributionCache; maxiters=cache.prob.maxiters, abstol=cache.prob.abstol, kwargs...)
    fill!(cache.distribution, inv(prod(size(cache.distribution))))
    sol = solve!(cache, cache.distribution, cache.prob.solver; maxiters, abstol, kwargs...)

    ModelSolution(cache, sol.nsteps, sol.resid)
end

function (c::ErgodicDistributionCache)(du,d0)
    u0 = reshape(d0, length.(c.grid))
    distribution!(c.tmp, u0, c.policies, c.grid)
    itensor(du, c.inds1) .= *(itensor(c.tmp, c.inds0), c.weights...)
    du .-= d0
end

function step!(c::ErgodicDistributionCache)
    distribution!(c.tmp, c.distribution, c.policies, c.grid)
    itensor(c.distribution, c.inds1) .= *(itensor(c.tmp, c.inds0), c.weights...)
end

function distribution!(dist, weight, policies::NTuple{N,F}, grid) where {N,F}
    C1 = CartesianIndices(length.(grid[N+1:end]))
    grid0 = ntuple(i->grid[i],N)
    @inbounds _distribution!(grid0, dist, policies, weight, C1)
end

function _distribution!(grid0, dist, policies, weight, chunk)
    for c1 in chunk
        C0 = CartesianIndices(length.(grid0))
        for c0 in C0
            t = (Tuple(c0)..., Tuple(c1)...)
            dist[t...] = 0.0
        end
        for c0 in C0
            t = (Tuple(c0)..., Tuple(c1)...)
            x = ntuple(i-> policies[i][t...], length(policies))
            w = weight[t...]
            __distribution!(grid0, dist, x, w, Tuple(c1))
        end
    end
end

function __distribution!(grid0::NTuple{M,S}, dist, x, ω, c1) where {M,S}
    start = ntuple(M) do i
        if x[i] ≤ first(grid0[i])
            1
        elseif x[i] ≥ last(grid0[i])
            length(grid0[i])-1
        else
            searchsortedfirst(grid0[i], x[i])-1
        end
    end
    wl = ntuple(M) do i
        l, r = (grid0[i][start[i]], grid0[i][start[i]+1])
        return (r-x[i])/(r-l)
    end
    wr = ntuple(M) do i
        l, r = (grid0[i][start[i]], grid0[i][start[i]+1])
        return (x[i]-l)/(r-l)
    end
    weights = (wl, wr)
    for cidx in CartesianIndices(ntuple(Returns(2), M))
        w = ntuple(i->weights[cidx[i]][i], M)
        c0 = ntuple(i->start[i]+(cidx[i]-1), M)
        dist[c0..., c1...] += ω*prod(w)
    end
end
