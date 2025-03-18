@kwdef struct ValueFunctionCache
    prob
    parameters
    grid = (prob.grid...,)
    policies = stack(Base.product(grid...)) do s
        prob.u0(s,parameters)
    end |> x -> ntuple(size(x)[1]) do j
        collect(selectdim(x,1,j))
    end
    residuals = if prob isa ValueFunction
        ()
    else
        (similar.(policies),)
    end
    values = zeros(prod(length.(grid)), length(prob.itpflags))
    itp = ntuple(size(values)[2]) do j
        _interpolate(
            grid,
            zeros(length.(grid)),
            prob.itpflags[j]
        )
    end
end
init(prob::T, parameters) where {T<:ValueFunctionType} =
    ValueFunctionCache(; prob, parameters)

function solve!(c::ValueFunctionCache, args...)

    init!(c.prob, c.values, c.grid, c.policies, c.residuals..., c.itp, c.parameters, args)

    solve!(vec(c.values), c.prob.solver; c.prob.kwargs...) do dv,v

        step!(c.prob, dv, v, c.grid, c.policies, c.residuals..., c.itp, c.parameters, (c.residuals..., args...))
        
        solve_howard!(dv, c.prob.f, c.grid, c.policies, c.itp, c.parameters,
            (c.residuals..., args...))

        dv .-= v
        
    end
    
end
