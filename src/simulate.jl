using ConcreteStructs, Compat

@kwdef @concrete struct Simulation
    f
    ss_cache
    x0
    T
end
@kwdef @concrete struct SimulationCache
    prob
    paths
    rules
    T
end

function step!(cache::SimulationCache{N})
    (; grid, paths, rules) = cache
    for i in 1:N
        Random.rand!(paths[i], 1:length(grid[i]))
    end
    paths .= rand(size(paths))
    paths!.((cache.paths,), cache.rules, cache.grid, (1:length(paths),))
end

function init(prob::Simulation)
    grid = ntuple(i-> collect(prob.grid[i]), length(prob.grid))
    paths = vcat(transpose.(_paths.(prob.rules, prob.x0, (prob.T,)))...)
    for i in 1:size(paths)[1]
        paths[i,1] = prob.x0[i]
    end
    rules = (r for r in prob.rules if r isa Function)
    SimulationCache(grid, paths, rules, prob.T)
end

function init!(cache::SimulationCache{N,F}) where {N,F}
    (; grid, rules, paths, T) = cache
    for t in 1:T-1
        for (i,rule) in enumerate(rules)
            @inbounds paths[i,t+1] = rule(paths,i,t)
        end
    end
    s
end

_paths(rule,s0,T) = zeros(Int64,T)
function _paths(rule::Vector{F},s0,T) where F
    Π = cumsum(rule)
    X = map(rand(T)) do x
        searchsortedfirst(Π, x)
    end
end
function _paths(rule::Matrix{F},s0,T) where F
    X, states = (rand(T), zeros(Int64, T))
    Π = cumsum(rule; dims=2)
    states[1] = s0
    for t in 2:T
        @views states[t] = searchsortedfirst(Π[states[t-1],:], X[t])
    end
    states
end

function simulate(n,T)
    d = rand.(n[1],n[1])
    d .= d ./ sum(d; dims=2)
    grid = collect.((1:n[1], 1:n[2]))
    f = (paths,i,t) -> 2
    s = init(Simulation(grid, (d,f), (2,1), T))
end
