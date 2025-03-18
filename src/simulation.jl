@kwdef @concrete struct Simulate
    f = (x1,x0,t) -> ()
    grid
    nature
    rules
    paths
end

@kwdef @concrete struct SimulationCache
    x0
    prob
    T
    shocks = OffsetArray(rand(Float64, length(prob.nature), T+1), 0, -1)
    states = OffsetArray(Matrix{eltype(x0)}(undef, length(x0), T+1), 0, -1)
    paths = OffsetArray(Matrix{Float64}(undef, length(prob.paths), T+1), 0, -1)
end
init(s::Simulate, x0, T) = SimulationCache(; prob=s, x0, T)
Base.getindex(s::SimulationCache, idx...) = s.states[idx...]
Base.setindex!(s::SimulationCache, x::Int, idx...) = (s.states[idx...] = x)
function Base.setindex!(s::SimulationCache, x::Float64, i, t)
    grid = s.prob.grid[i]
    s.states[i,t] = if x ≤ first(grid)
        1
    elseif x ≥ last(grid)
        length(grid)
    else
        idx = searchsortedfirst(grid,x)
        δ = (x-grid[idx-1])/(grid[idx]-grid[idx-1])
        δ < 0.5 ? idx-1 : idx
    end
end

function solve!(s::SimulationCache)

    @views s.states[:,0] .= s.x0
    Random.rand!(s.shocks)
    for t in 0:s.T-1
        
        for (i,key) in enumerate(keys(s.prob.nature))
            x = s.states[key,t]
            ϵ = s.shocks[i,t]
            s[key,t+1] = simulate_nature(s.prob.nature[key], x, ϵ)
        end

        for key in keys(s.prob.rules)
            x = s.states[key,t]
            s[key,t+1] = simulate_rules(s.prob.rules[key], x, t)
        end
        s.prob.f(s,s.states,t)
        
    end

    for t in 1:s.T

        for (i,f) in enumerate(s.prob.paths)

            s.paths[i,t] = f(s.states,t)
            
        end

    end
    
end
simulate_nature(π::Vector, x, ϵ) = searchsortedfirst(π,ϵ)
simulate_nature(π::Matrix, x, ϵ) = @views searchsortedfirst(π[x,:],ϵ)
simulate_rules(f::F, x, t) where {F<:Function} = f(x,t)
simulate_rules(A::Vector, x, t) = A[t]
simulate_rules(A::Array, x, t) = A[CartesianIndex(x)]


