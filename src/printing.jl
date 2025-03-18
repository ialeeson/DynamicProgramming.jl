# ### Print Value Function Solution
# function Base.show(io::IO, r::ValueFunctionSolution)
#     grid = r.cache.grid
#     _n = collect(length.(grid))
#     _range = [(floor(Int,x),ceil(Int,y))
#              for (x,y) in zip(minimum.(grid),maximum.(grid))]
#     @printf io "\nResults of Value Function Iteration\n"
#     @printf io " * Converged: %s\n" r.converged
#     @printf io " * Number of Knots: %s \n" _n
#     @printf io " * Grid: %s \n" _range
#     @printf io " * Iterations: %d\n" r.nsteps
#     @printf io " * Time: %.2f s\n" r.time
#     @printf io " * Error: %e\n" r.resid
# end

# function Base.show(io::IO, s::EconomicModelSolution)
#     (; grid, converged, nsteps, reisd, time) = s
#     @printf io "\nResults\n"
#     @printf io " * Converged: %s\n" converged
#     @printf io " * Iterations: %d\n" nsteps
#     @printf io " * Time: %.2f s\n" time
#     @printf io " * Error: %e\n" resid
# end

# ### Print Value Function Step
# function Base.show(io::IO, r::ShootingMethodCache)
#     @printf io "%d Steps; " r.nsteps
#     @printf io "Err: %e\n" r.resid
#     @printf io "%d Steps; " r.ssnsteps
#     @printf io "Err: %e\n" r.ssresid
# end

# function Base.show(io::IO, c::Union{ValueFunctionCache, EulerEquationCache})
#     @printf io "%d Steps; " c.nsteps
#     @printf io "Err: %e\n" c.resid
# end

# function Base.show(io::IO, r::SteadyStateCache)
#     @printf io "Agg: %s\n" r.aggregates
#     for (nsteps,resid) in zip(r.model_cache.nsteps, r.model_cache.resid)
#         @printf io "%d Steps; " nsteps
#         @printf io "Err: %e\n" resid
#     end
# end
# function Base.show(io::IO, r::ShootingMethodCache)
#     @printf IOContext(io, :compact => true) "Agg: %s\n" r.aggregates[:,3]
# end

# function Base.show(io::IO, sol::ModelSolution)
#     for i in 1:1
#         @printf io "%d Steps; " sol.nsteps[i]
#         @printf io "Err: %e\n" sol.resid[i]
#     end
# end

# function Base.show(io::IO, c::EconomicModelCache)
#     for (nsteps,resid) in zip(c.nsteps, c.resid)
#         @printf io "%d Steps; " nsteps
#         @printf io "Err: %e\n" resid
#     end
# end

function Base.show(io::IO, s::IterationSolution)
    for (nsteps, resid) in zip(s.nsteps, s.resid)
        @printf io "\n%d Steps; " nsteps
        @printf io "Err: %e" resid
    end
end
