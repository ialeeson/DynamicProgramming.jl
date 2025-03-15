### Problem
@kwdef @concrete struct EconomicModelProblem
    grid
    policyprobs
    distprob = missing
end

### Cache
@kwdef @concrete struct EconomicModelCache
    prob
    parameters
    grid = (collect.((prob.grid...,))...,)
    caches = ntuple(i->init(prob.policyprobs[i], parameters),
        length(prob.policyprobs))
    averages = if prob.distprob isa Missing
        missing
    else
        ntuple(i->zeros(length(grid) - length(prob.distprob.weights)),
            length(prob.policyprobs))
    end
    distribution_caches = if prob.distprob isa Missing
        missing
    else
        ntuple(i->init(prob.distprob, grid, caches[i].policies),
            length(prob.policyprobs))
    end
    nsteps = zeros(Int64, length(caches))
    resid = zeros(length(caches))
end
init(prob::EconomicModelProblem, parameters) =
    EconomicModelCache(; prob, parameters)

### Methods
function solve!(c::EconomicModelCache)
    for i in 1:length(c.caches)
        sol = solve!(c.caches[i])
        # c.nsteps[i] = sol.nsteps
        # c.resid[i] = sol.resid
    end
    if !(c.prob.distprob isa Missing)
        solve!.(c.distribution_caches)
        copyto!.(c.averages, _averages.(c.distribution_caches))
    end
    c
end

function _averages(distribution_cache)
    (; inds0, grid, distribution, policies) = distribution_cache
    ntuple(length(policies)) do i 
        *(itensor(distribution, inds0),
            itensor(grid[i], inds0[i]),
            δ(inds0[begin:i-1]..., inds0[i+1:end]...))[1]
    end
end
