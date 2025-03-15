### Problem
@kwdef @concrete struct SteadyState
    f
    x0
    model_probs
    maxiters = 10^2
    abstol = 1e-10
    kwargs = (; )
    solver = Iterative(λ)
end

### Cache
@kwdef @concrete struct SteadyStateCache
    prob
    parameters
    model_caches = ntuple(i->init(prob.model_probs[i], parameters),
        length(prob.model_probs))
    aggregates = copy(prob.x0)
    # values = ntuple(i->model_cache.cache[i].values, length(model_cache.cache))
    # distribution = zeros(length.((model_cache.prob.grid...,)))
    # caches = ntuple(length(prob.prob)) do i
    #     (; grid, parameters) = model_cache.prob
    #     (; policies) = model_cache
    #     init(prob.prob[i]; grid=(grid...,), parameters,
    #         policies, distribution, aggregates)
    # end
end
init(prob::SteadyState, parameters) = SteadyStateCache(; prob, parameters)

function (ss::SteadyStateCache)(dx,x)
    ss.parameters(x)
    for cache in ss.model_caches
        solve!(cache)
    end
    avg = ntuple(i->ss.model_caches[i].averages, length(ss.model_caches))
    ss.prob.f(dx, x, avg)
end

function solve!(ss::SteadyStateCache; maxiters=ss.prob.maxiters, abstol=ss.prob.abstol)
    @printf "\nSteady State:\n"
    solve!(ss, ss.aggregates, ss.prob.solver; verbose=true, maxiters, abstol, ss.prob.kwargs...)

end
