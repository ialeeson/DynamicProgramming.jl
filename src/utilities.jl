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
