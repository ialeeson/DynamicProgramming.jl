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
(f::FlipSign)(x...) = -f.f(x...)

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
    g = FlipSign(Fix2(f,p))
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
        DifferentiationInterface.SecondOrder(
            Optimization.AutoForwardDiff(),
            Optimization.AutoForwardDiff()
        );
        cons
    ) do u,p
        -f(u,p)
    end
    prob = OptimizationProblem(fun, u0, p;
        lcons, ucons, lb, ub)
    sol = solve(prob, solver)
    if isnan(sol.objective)
        (; u=u0, v=f(u0,p))
    else
        (; u=sol.u, v=-sol.objective)
    end
end

function inner_solve(ee::F, solver::T; kwargs, p)  where {F,T<:BracketingNonlinearSolve.AbstractBracketingAlgorithm}
    (; lb, ub) = kwargs
    eelb, eeub = (ee(lb,p), ee(ub,p))
    if eelb > 0.0
        (; u=lb, v=eelb)
    elseif eeub < 0.0
        (; u=ub, v=eeub)
    else
        prob = IntervalNonlinearProblem{false}(ee,(lb[1],ub[1]),p)
        sol = solve(prob, solver)
        (; u=sol.u, v=sol.resid)
    end
end

function dispatch_multi!(f!::F, res, args...; l) where F
    chunks = Iterators.partition(1:l, div(l, Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn f!(res, args..., chunk)
    end
    fetch.(tasks)
end

function _interpolate!(itp, v; sz)
    l = prod(sz)
    for (i,itp) in enumerate(itp)
        @views __interpolate!(itp, v[1+(i-1)*l:i*l], sz)
    end
end
