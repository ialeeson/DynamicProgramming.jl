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

