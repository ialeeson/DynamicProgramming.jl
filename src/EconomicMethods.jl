module EconomicMethods

using Compat, ConcreteStructs, Printf, Adapt, Random
using StaticArrays, OffsetArrays
using Interpolations, ITensors
using NonlinearSolve, Optim, OptimizationOptimJL, DifferentiationInterface, CommonSolve
using Makie, KernelAbstractions

import Base.@kwdef, Base.Fix1, Base.Fix2, Compat.Fix, Base.copy
import CommonSolve.solve!, CommonSolve.step!, CommonSolve.init, CommonSolve.solve
import Statistics.quantile, Random.rand!

abstract type ValueFunctionType end

include("grids.jl")
include("flags.jl")
include("utilities.jl")
include("value_function_problem.jl")
include("howard_policy_iteration.jl")
include("naive_euler_equation_iteration.jl")
include("euler_equation_iteration.jl")
include("value_function_iteration.jl")
include("policy_problem.jl")
include("iteration.jl")

include("rowenhorst.jl")
include("printing.jl")
include("simulation.jl")


export EulerEquation, ValueFunction, HowardEquation, PolicyProblem, NaiveEulerEquation
export Simulate
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
