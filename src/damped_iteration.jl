struct Iteration
    
end
    
    while cache.resid > abstol && cache.nsteps < maxiters
        step!(cache)
        (cache.nsteps == 1 || mod(cache.nsteps,10) == 0) && @info cache
    end
    converged = cache.resid < abstol
end

function CommonSolve.solve(
        prob::IntervalNonlinearProblem, alg::DampedIteration, args...;
        maxiters = 1000, abstol = nothing, verbose::Bool = true, kwargs...
)
    @assert !SciMLBase.isinplace(prob) "`Bisection` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right)))

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right
        )
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right
        )
    end

    if sign(fl) == sign(fr)
        verbose &&
            @warn "The interval is not an enclosing interval, opposite signs at the \
                   boundaries are required."
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right
        )
    end

    i = 1
    while i ≤ maxiters

        step!(cache)
        
        if abs((right - left) / 2) < abstol
            return SciMLBase.build_solution(
                prob, alg, mid, fm; retcode = ReturnCode.Success, left, right
            )
        end

        i += 1
    end

    sol !== nothing && return sol

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right
    )
end

function CommonSolve.solve(
        prob::IntervalNonlinearProblem, alg::ITP, args...;
        maxiters = 1000, abstol = nothing, verbose::Bool = true, kwargs...
)
    @assert !SciMLBase.isinplace(prob) "`ITP` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right
        )
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right
        )
    end

    if sign(fl) == sign(fr)
        verbose &&
            @warn "The interval is not an enclosing interval, opposite signs at the \
                   boundaries are required."
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right
        )
    end

    ϵ = abstol
    k2 = alg.k2
    k1 = alg.scaled_k1 * abs(right - left)^(1 - k2)
    n0 = alg.n0
    n_h = ceil(log2(abs(right - left) / (2 * ϵ)))
    mid = (left + right) / 2
    x_f = left + (right - left) * (fl / (fl - fr))
    xt = left
    xp = left
    r = zero(left) # minmax radius
    δ = zero(left) # truncation error
    σ = 1.0
    ϵ_s = ϵ * 2^(n_h + n0)

    i = 1
    while i ≤ maxiters
        span = abs(right - left)
        r = ϵ_s - (span / 2)
        δ = k1 * span^k2

        x_f = left + (right - left) * (fl / (fl - fr))  # Interpolation Step

        diff = mid - x_f
        σ = sign(diff)
        xt = ifelse(δ ≤ diff, x_f + σ * δ, mid)  # Truncation Step

        xp = ifelse(abs(xt - mid) ≤ r, xt, mid - σ * r)  # Projection Step

        if abs((left - right) / 2) < ϵ
            return SciMLBase.build_solution(
                prob, alg, xt, f(xt); retcode = ReturnCode.Success, left, right
            )
        end

        # update
        tmin, tmax = minmax(xt, xp)
        xp ≥ tmax && (xp = prevfloat(tmax))
        xp ≤ tmin && (xp = nextfloat(tmin))
        yp = f(xp)
        yps = yp * sign(fr)
        T0 = zero(yps)
        if yps > T0
            right, fr = xp, yp
        elseif yps < T0
            left, fl = xp, yp
        else
            return SciMLBase.build_solution(
                prob, alg, xp, yps; retcode = ReturnCode.Success, left, right
            )
        end

        i += 1
        mid = (left + right) / 2
        ϵ_s /= 2

        if Impl.nextfloat_tdir(left, prob.tspan...) == right
            return SciMLBase.build_solution(
                prob, alg, right, fr; retcode = ReturnCode.FloatingPointLimit, left, right
            )
        end
    end

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right
    )
end


function solve!(cache::EconomicModelCache; kwargs...)

    for key in keys(kwargs)
        setfield!(cache, key, kwargs[key])
    end
    for (i,c) in enumerate(cache.cache)
        c.maxiters = cache.maxiters[i]
        c.abstol = cache.abstol[i]
    end
    solve!.(cache.cache)
    cache.prob.callback(cache)
    
end

### Init
function init!(cache::T) where {T<:ModelCache}
    
    (; prob, grid, values, policies, parameters, itp) = cache

    chunks = Iterators.partition(CartesianIndices(length.(grid)),
        div(prod(length.(grid)), Threads.nthreads()))
    tasks = map(chunks) do chunk
        Threads.@spawn _init!(prob, grid, values, policies, parameters, chunk)
    end
    fetch.(tasks)
    _interpolate!.(itp, values)
    cache.nsteps = 0
    cache.resid = Inf

    return cache
    
end
_init!(prob, grid, values, policies, parameters, chunk) = ()

### Step
function step!(cache::T; nsteps=1, howard=cache.prob.howard) where {T<:ModelCache}

    (; prob, grid, values, policies, parameters, itp, lower, upper) = cache
    
    chunks = Iterators.partition(CartesianIndices(length.(grid)),
        div(prod(length.(grid)), Threads.nthreads()))
    for steps in 1:nsteps
        tasks = map(chunks) do chunk
            Threads.@spawn _step!(prob, grid, values, policies, parameters, itp, lower, upper, chunk)
        end
        cache.nsteps += 1
        cache.resid = sum(fetch.(tasks))
        _interpolate!.(itp, values)
        prob.callback(cache)
    end
    solve!(howard, cache)
    
    return cache
    
end
_step!(prob, grid, values, policies, parameters, itp, lower, upper, chunk) = ()

### Solve
function solve!(cache::T; howard=cache.prob.howard) where {T<:ModelCache}
    
    init!(cache)
    
    (; maxiters, abstol) = cache
    while cache.resid > abstol && cache.nsteps < maxiters
        step!(cache; howard)
    end
    cache
    converged = cache.resid < abstol
    
    return cache
    
end
