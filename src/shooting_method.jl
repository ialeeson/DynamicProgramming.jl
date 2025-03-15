### Problem
@kwdef @concrete struct ShootingMethodProblem
    ss_prob
    f
    T = 500
    abstol = 1e-10
    maxiters = 10^2
    kwargs = (;)
    solver = Iteration()
end

### Cache
@kwdef @concrete mutable struct ShootingMethodCache
    prob
    parameters
    ss_cache = init(prob.ss_prob, parameters)
    model_cache = init(
        prob.ss_prob.model_prob.prob[end];
        grid = (ss_cache.model_cache.prob.grid...,),
        parameters,
        policies = ss_cache.model_cache.policies
    )
    ss_distribution = copy(ss_cache.distribution)
    policies = ntuple(i -> similar.(ss_cache.model_cache.policies), prob.T+1)
    aggregates = zeros(length(ss_cache.aggregates), prob.T+1)
end
init(prob::ShootingMethodProblem, parameters) =
    ShootingMethodCache(; prob, parameters)

### Step
function _backward_step!(c::ShootingMethodCache, u, t)
    c.prob.f(t,c.model_cache.parameters)
    @views c.model_cache.parameters(u)
    if t == c.prob.T+1
        solve!(c.model_cache; verbose=false)
    else
        step!(c.model_cache)
    end
    copyto!.(c.policies[t], c.model_cache.policies)
end

function _forward_step!(c::ShootingMethodCache, du, t)
    if t == 2
        copyto!(c.ss_cache.distribution, c.ss_distribution)
    else
        copyto!.(c.model_cache.policies, c.policies[t-1])
        for cache in c.ss_cache.cache
            step!(cache)
        end
    end
    inds = Index.(length.(c.model_cache.grid))
    for i in 1:length(du)
        du[i] = sum(
            *(
                itensor(c.ss_cache.distribution, inds),
                itensor(collect(c.model_cache.grid[i]), inds[i])
            )
        )
    end
end

function (c::ShootingMethodCache)(du,u)
    for t in c.prob.T+1:-1:2
        @views _backward_step!(c, u[:,t-1], t)
    end
    for t in 2:c.prob.T+1
        @views _forward_step!(c, du[:,t-1], t)
    end
    @views du .-= u
end

### Init
function init!(c::ShootingMethodCache)
    c.prob.f(1, c.model_cache.parameters)
    solve!(c.ss_cache)
    copyto!(c.ss_distribution, c.ss_cache.distribution)
    copyto!.(c.policies[1], c.ss_cache.model_cache.policies)
    copyto!.(eachslice(c.aggregates; dims=2), (c.ss_cache.aggregates,))
end

function solve!(cache::ShootingMethodCache; maxiters=cache.prob.maxiters, abstol=cache.prob.abstol)
    init!(cache)
    @printf "\nShooting Method:\n"
    @views solve!(cache, cache.aggregates[:,2:end], cache.prob.solver; verbose=true, maxiters, abstol, cache.prob.kwargs...)
    @info cache
end
