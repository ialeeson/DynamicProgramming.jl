using ConcreteStructs, Printf, Setfield, StaticArrays, Random, OffsetArrays
using GLMakie, NonlinearSolve, Interpolations, ITensors, LinearMaps
using Optim, OptimizationOptimJL, DifferentiationInterface
import Base.@kwdef, Base.Fix1, Base.Fix2, Compat.Fix, Base.copy
import CommonSolve.solve!, CommonSolve.step!, CommonSolve.init, CairoMakie.plot, CommonSolve.solve
import Statistics.quantile

abstract type Grid end

struct Markov{I,F} <: Grid
    grid::I
    weights::Matrix{F}
end

### InterpolationTypes
__interpolate(grid, values, flags) =
    extrapolate(scale(interpolate(values, flags), grid), Interpolations.Flat())

_interpolate(grid, values, flags) = __interpolate(grid, values, flags)
function _interpolate(grid, values, flags::Tuple)
    grid = ntuple(length(flags)) do i
        if flags[i] isa NoInterp
            range(1,length(grid[i]),length(grid[i]))
        else
            grid[i]
        end
    end
    __interpolate(grid, values, flags)
end
_interpolate(grid, values, flags::Gridded) =
    extrapolate(interpolate(grid, values, flags), Linear())

_interpolate!(itp, A, sz) = _interpolate!(
    Interpolations.itpflag(itp),
    Interpolations.coefficients(itp),
    A,
    sz
)
_interpolate!(itp, coefs, A, sz) = copyto!(coefs,A)
_interpolate!(itp::Interpolations.Extrapolation, A, sz) =
    _interpolate!(itp.itp, A, sz)
function _interpolate!(flags, coefs::T, A::R, sz) where {T<:Interpolations.OffsetArray,R<:AbstractArray}

    for (i,cidx) in enumerate(CartesianIndices(sz))
        coefs[cidx] = A[i]
    end
    for j in 1:length(size(coefs))
        for i in minimum(axes(coefs,j)):0
            selectdim(coefs,j,i) .= 0.0
        end
        for i in sz[j]+1:maximum(axes(coefs,j))
            selectdim(coefs,j,i) .= 0.0
        end
    end
    Interpolations.prefilter!(eltype(coefs), coefs, flags)
    
end

### Weighted
struct Weighted
    pre
    post
    weights
    itpflags
    Weighted(pre, post, weights, flags...) = begin
        for v in values(weights)
            !(all(sum(v; dims=2) .≈ 1)) &&
            error("Weight matrix is not valid.")
        end
        new(pre, post, weights, flags)
    end
end
@kwdef @concrete struct WeightedInterpolation
    pre
    post
    itp
    weights
    dims
    tmp0
    tmp1
end
@inline (i::WeightedInterpolation)(x...) = i.itp(x...)

function _interpolate(grid, _values, flags::Weighted)
    pre = flags.pre
    post = flags.post
    itp = _interpolate(grid, _values, flags.itpflags)
    dims = (keys(flags.weights)...,)
    tmp0 = itensor(
        similar(_values),
        Index.(length.(grid))
    )
    tmp1 = itensor(
        similar(_values),
        ntuple(length(grid)) do i
            i in dims ? Index(length(grid[i])) : inds(tmp0)[i]
        end
    )
    weights = Tuple(
        itensor(flags.weights[k], inds(tmp1)[k], inds(tmp0)[k])
        for k in keys(flags.weights)
    )
    WeightedInterpolation(; pre, post, itp, weights, dims, tmp0, tmp1)
end
function _interpolate!(witp::WeightedInterpolation, A, sz)
    (; pre, post, itp, tmp0, tmp1, weights) = witp
    map!(pre, ITensors.tensor(tmp0), A)
    tmp1 .= post.(*(tmp0, weights...))
    _interpolate!(itp, ITensors.tensor(tmp1), sz)
end


### Integral
@kwdef struct Integral
    weights
    nodes = 1:size(weights)[1]
    itpflags
end

struct IntegralInterpolation{W,N,IT}
    nodes::N
    weights::W
    itp::IT
end
(t::IntegralInterpolation)(f,j) = sum(
    t.itp(f(i,x)...)*t.weights[i] for (i,x) in enumerate(t.nodes)
)
(t::IntegralInterpolation{Matrix{F}})(f,j) where {F} = sum(
    t.itp(f(i,x)...)*t.weights[j,i] for (i,x) in enumerate(t.nodes)
)
function _interpolate(grid, values, int::Integral)
    (; nodes, weights, itpflags) = int
    itp = _interpolate(grid, values, itpflags)
    IntegralInterpolation(nodes, weights, itp)
end
_interpolate!(itp::IntegralInterpolation, A, sz) = _interpolate!(itp.itp, A, sz)

import Interpolations.interpolate!, Base.map!

### States
struct State{S,T}
    s::T
    cidx::CartesianIndex{S}
end
Base.getindex(S::State,i) = S.s[i]

## FlipSign
struct FlipSign{F} <: Function
    f::F
    FlipSign(f::F) where {F} = new{F}(f)
    FlipSign(f::Type{F}) where {F} = new{Type{F}}(f)
end
(f::FlipSign)(x) = -f.f(x)

function diff!(x,y,f)
    if length(CartesianIndices(y)) ≤ Threads.nthreads()
        resid = 0.0
        @inbounds for cidx in CartesianIndices(y)
            resid += abs(x[cidx] - y[cidx])
            x[cidx] = y[cidx]
        end
        return resid
    else
        chunks = Iterators.partition(CartesianIndices(y),
            div(length(CartesianIndices(y)), Threads.nthreads()))
        tasks = map(chunks) do chunk
            Threads.@spawn let resid = 0.0
                @inbounds for cidx in chunk
                    resid += f(x[cidx] - y[cidx])
                    x[cidx] = y[cidx]
                end
                resid
            end
        end
        sum(fetch.(tasks))
    end
end

function inner_solve(f::F, solver::T; kwargs, p)  where {F,T<:Optim.UnivariateOptimizer}
    g = Fix2(f,p)
    (; lb, ub) = kwargs
    if ub ≤ lb
        (; u=lb, v=-g(lb))
    else 
        sol = optimize(g, lb[1], ub[1], solver)
        (; u=sol.minimizer, v=-sol.minimum)
    end
end

function inner_solve(f::F, solver::T; kwargs, p)  where {F,T<:Optim.IPOptimizer}
    (; cons, lcons, ucons, lb, ub, u0) = kwargs
    fun = OptimizationFunction(
        f,
        DifferentiationInterface.SecondOrder(
            Optimization.AutoForwardDiff(),
            Optimization.AutoForwardDiff()
        );
        cons
    )
    prob = OptimizationProblem(fun, u0, p;
        lcons, ucons, lb, ub)
    sol = solve(prob, solver)
    if isnan(sol.objective)
        (; u=u0, v=-f(u0,p))
    else
        (; u=sol.u, v=-sol.objective)
    end
end

function inner_solve(ee::F, solver::T; kwargs, p)  where {F,T<:BracketingNonlinearSolve.AbstractBracketingAlgorithm}
    (; lb, ub) = kwargs
    eelb, eeub = (ee(lb,p), ee(ub,p))
    (; u, r) = if eelb < 0.0
        (; u=lb, r=eelb)
    elseif eeub > 0.0
        (; u=ub, r=eeub)
    else
        prob = IntervalNonlinearProblem{false}(ee,(lb[1],ub[1]),p)
        sol = solve(prob, solver)
        (; u=sol.u, r=sol.resid)
    end
end


@inline fv(u::N, s, itp) where {N<:Number} = itp[1](u, s[2:end]...)
@inline fv(u::SVector{N,T}, s, itp) where {N,T} =
            ntuple(j->itp[j](u..., s[N+1:end]...), length(itp))
@inline fv(u::NTuple{N,T}, s, itp) where {N,T} =
            ntuple(j->itp[j](u..., s[N+1:end]...), length(itp))

# struct ValueFunctionProblem
# end

# struct ValueFunctionCache
# end

# struct ValueFunctionSolution
# end

# function CommonSolve.solve(cache::ValueFunctionCache, alg::Iteration;
#     maxiters=10^2, abstol=nothing, verbose::Bool=true)

#     for steps in 1:maxiters
#         f = Fix{3}(f, parameters)
#         step!(f,u2,u1,s,v)
#         interpolate!(itp,v)
#     end
#     build_solution()
    
# end

# function inner_solve!(solver::T, f::F, g::H, u, v, state) where {T<:BracketingNonlinearSolve.AbstractBracketingAlgorithm,F}
#     cidx = state.cidx
#     u0 = ntuple(i->policies[cidx], length(policies))
#     lb = lower(minimum.(grid), s, parameters)
#     ub = upper(maximum.(grid), s, parameters)
#     prob = NonlinearProblem{false}((lb,ub),p) do u,p
#         f(u,s,v,p)
#     end
#     sol = solve(prob, solver)
# end

# function inner_solve!(solver::Optim.UnivariateOptimizer, f::F, s, v, p; lb, ub, kwargs...) where F
#     Fix{3}(Fix{4}(f,P),u)
#     sol = optimize(lb[1], ub[1], solver) do u
#         v = ntuple(i-> values[i](u..., cidx[1]...), length(values))
#         -f(v,u,s,p)
#     end
#     (; u=sol.minimizer, v=-sol.minimum)
# end

# function step!(cache::ValueFunctionCache, values, policies, chunk)
#     (; solver, policies, values) = cache
#     for cidx in chunk
#         state = State(ntuple(i->grid[cidx[i]], length(cidx)), cidx)
#         f = Fix{3}(Fix{4}(cache.f, parameters), values)
#         inner_solve!(solver, f, policies, values, state)
#     end
# end

# function step!(cache::HowardPolicyCache, values, chunk)
#     (; solver, policies, itp) = cache
#     for cidx in chunk
#         state = State(ntuple(i->grid[cidx[i]], length(cidx)), cidx)
#         u = ntuple(i->policies[i][cidx], length(policies))
#         for j in 1:length(f)
#             values[j][cidx] = f[j](u, state, itp, parameters)
#         end
#     end
# end

# function step!(cache::EulerEquationCache, values, policies, chunk)
#     (; solver, policies, values) = cache
#     for cidx in chunk
#         state = State(ntuple(i->grid[cidx[i]], length(cidx)), cidx)
#         ee = Fix{3}(Fix{4}(cache.ee, parameters), values)
#         f = Fix{3}(Fix{4}(cache.f, parameters), values)
#         inner_solve!(solver, ee, f, policies, values, state)
#     end
# end

@kwdef struct ValueFunctionCache{P}
    prob::P
    parameters
    grid = (prob.grid...,)
    policies = stack(Base.product(grid...)) do s
        prob.u0(s,parameters)
    end |> x -> ntuple(j->collect(selectdim(x,1,j)), size(x)[1])
    values = zeros(prod(length.(grid)), length(prob.itpflags))
    itp = ntuple(size(values)[2]) do j
        _interpolate(
            (prob.grid...,),
            zeros(length.(grid)),
            prob.itpflags[j]
        )
    end
    residuals = similar.(policies)
end

function solve!(c::ValueFunctionCache, args...)
    init!(c.prob, c.values, c.grid, c.policies, c.residuals, c.itp, c.parameters, args)
    solve!(vec(c.values), c.prob.solver; maxiters=c.prob.maxiters, abstol=c.prob.abstol) do dv,v
        step!(c.prob, dv, v, c.grid, c.policies, c.residuals, c.itp, c.parameters, args)
        if c.prob.policy_prob isa Missing
            dv .-= v
        else
            sol = solve!(dv, c.prob.policy_prob.solver; maxiters=c.prob.policy_prob.maxiters, abstol=c.prob.policy_prob.abstol) do du,u
                step!(c.prob.policy_prob, du, u,
                    c.grid, c.policies, c.residuals, c.itp, c.parameters, args)
                du .-= u
            end
            @. dv = sol.u - v
        end
    end
end

function dispatch_multi!(f!::F, res, args...; l) where F
    chunks = Iterators.partition(1:l, div(l, Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn f!(res, args..., chunk)
    end
    fetch.(tasks)
end

@kwdef @concrete struct Howard
    f
    solver
    maxiters = 10^2
    abstol = 1e-10
end

function _step!(dv, prob::Howard, grid, policies, residuals, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = ntuple(j->policies[j][cidx], length(policies))
        v = prob.f(itp, u0, state, parameters, args...)
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
    end
    
end

@kwdef @concrete struct EulerEquation
    f
    grid
    ee
    u0
    v0
    itpflags
    solver = missing
    inner_solver = missing
    policy_prob = missing
    kwargs = Returns(())
    maxiters = 10^2
    abstol = 1e-10
end

function _step!(dv, prob::EulerEquation, grid, policies, residuals, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(i -> policies[i][cidx], length(policies)))
        kwargs = prob.kwargs(u0, state, parameters, args...)
        (; u, r) = inner_solve(prob.inner_solver; kwargs, p=parameters) do u,p
            prob.ee(itp, u, state, p, args...)
        end
        v = prob.f(itp, u, state, parameters, args...)
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
        for j in eachindex(u)
            policies[j][cidx] = u[j]
            residuals[j][cidx] = r[j]
        end
    end
    
end

@kwdef @concrete struct ValueFunction
    f
    grid
    u0
    v0
    itpflags
    solver = missing
    inner_solver = missing
    policy_prob = Howard(; f=f, solver=Iteration(1.0))
    kwargs = Returns(())
    maxiters = 10
    abstol = 1e-10
end

init(prob::Union{EulerEquation, ValueFunction}, parameters) =
    ValueFunctionCache(; prob, parameters)

init!(prob::Union{ValueFunction,EulerEquation}, dv, grid, residuals, policies, itp, parameters, args) =
    dispatch_multi!(_init!, dv, prob, grid, policies, residuals, itp, parameters, args; l=prod(length.(grid)))

function step!(prob::Union{ValueFunction,Howard,EulerEquation}, dv, v, grid, policies, residuals, itp, parameters, args)
    l = prod(length.(grid))
    for (i,itp) in enumerate(itp)
        @views _interpolate!(itp, v[1+(i-1)*l:i*l], length.(grid))
    end
    resid = dispatch_multi!(_step!, dv, prob, grid, policies, residuals, itp, parameters, args; l=l)
end

function _init!(dv, prob::Union{ValueFunction,EulerEquation}, grid, policies, residuals, itp, parameters, args, chunk)

    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(j->policies[j][cidx], length(policies)))
        v = prob.v0(u0, state, parameters, args...)
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
    end
    
end

function _step!(dv, prob::ValueFunction, grid, policies, residuals, itp, parameters, args, chunk) 

    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(j->policies[j][cidx], length(policies)))
        kwargs = prob.kwargs(u0, state, parameters, args...)
        (; u, v)=inner_solve(prob.inner_solver; kwargs, p=parameters) do u,p
            -prob.f(itp, u, state, p, args...)
        end
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
        for j in eachindex(u)
            policies[j][cidx] = u[j]
        end
    end
    
end

### Iterative
@kwdef struct Iteration{F}
    λ::F = 1.0
end
@concrete struct IterationSolution
    u
    nsteps
    resid
end
solve!(f, u0, alg::Missing; kwargs...) = ()

function solve!(f, u0, alg::Iteration; verbose=false, maxiters=10^3, abstol=1e-10, kwargs...)
    resid, du = (Inf, copy(u0))
    steps = 0
    for i in 1:maxiters
        f(du,u0)
        u0 .+=  alg.λ .* du
        resid = mapreduce(abs2,+,du)
        resid < abstol && return IterationSolution(u0, i, resid)
        verbose && mod(steps,5) == 0 && begin
            @printf "\n%d Steps; " steps
            @printf "Err: %e\n" resid
        end
        steps += 1
    end
    IterationSolution(u0, steps, resid)
end

function solve!(f, u0, alg::NonlinearSolveBase.AbstractNonlinearSolveAlgorithm; kwargs...)

    prob = NonlinearProblem{true}(u0,()) do du,u,p
        f(du,u)
    end
    sol = solve(prob, alg)
    u0 .= sol.u
end
solve(p::NonlinearProblem{true}, alg::Iteration; kwargs...) =
    solve(p.f, p.u0, alg; kwargs...)
# solve(p::SteadyState, alg::Iteration; kwargs...) =
#     solve(p, p.aggregates, alg; kwargs...)


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
        rand() < δ ? idx : idx-1
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

