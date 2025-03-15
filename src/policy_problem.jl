@kwdef @concrete struct PolicyProblem
    probs
    solver = missing
    maxiters = 10^3
    abstol = 1e-20
end

@kwdef struct PolicyProblemCache
    prob
    parameters
    caches = ntuple(i->init(prob.probs[i], parameters), length(prob.probs))
    policies = broadcast(caches) do cache
        flags = ntuple(i->cache.grid[i] isa UnitRange ? NoInterp() : BSpline(Linear()), length(cache.grid))
        _interpolate.((cache.grid,), cache.policies, (flags,))
    end |> Tuple
    itp = ntuple(i->caches[i].itp, length(caches))
    nsteps = zeros(Int64, length(caches))
    resid = collect(Inf for i in 1:length(caches))
end
init(prob::PolicyProblem, parameters) =
    PolicyProblemCache(; prob, parameters)

function solve!(c::PolicyProblemCache)
    for (i,cache) in enumerate(c.caches)
        sol = solve!(cache, c.itp, c.policies)
        for (j,policies) in enumerate(c.policies[i])
            @views _interpolate!(policies, cache.policies[j],
                length.(cache.grid))
        end
        c.nsteps[i] = sol.nsteps
        c.resid[i] = sol.resid
    end
    IterationSolution(c, c.nsteps, c.resid)
end




