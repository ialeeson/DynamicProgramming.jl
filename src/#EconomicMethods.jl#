module EconomicMethods

using ConcreteStructs, Printf, Setfield, StaticArrays, Random, OffsetArrays
using GLMakie, NonlinearSolve, Interpolations, ITensors, LinearMaps
using Optim, OptimizationOptimJL, DifferentiationInterface
import Base.@kwdef, Base.Fix1, Base.Fix2, Compat.Fix, Base.copy
import CommonSolve.solve!, CommonSolve.step!, CommonSolve.init, CairoMakie.plot, CommonSolve.solve
import Statistics.quantile

include("grids.jl")
include("flags.jl")
include("utilities.jl")
include("value_function_problem.jl")
include("howard_policy_iteration.jl")
include("euler_equation_iteration.jl")
include("value_function_iteration.jl")
include("iteration.jl")
include("policy_problem.jl")

include("rowenhorst.jl")
include("printing.jl")
include("simulation.jl")


export EulerEquation, ValueFunction, Howard, PolicyProblem, Simulate
export solve!, step!, init, init!, plot
export Uniform, Rowenhorst, Weighted, Integral, Iteration
export rowenhorst_grid, rowenhorst_matrix
export FlipSign

end


# include("plot.jl")
# include("economic_model.jl")
# include("ergodic_distribution.jl")
# include("steady_state.jl")
# include("shooting_method.jl")
