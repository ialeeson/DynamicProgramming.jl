@kwdef struct ValueFunctionCache{P}
    prob::P
    parameters
    grid = (prob.grid...,)
    policies = stack(Base.product(grid...)) do s
        prob.u0(s,parameters)
    end |> x -> ntuple(size(x)[1]) do j
        adapt(prob.array_type, collect(selectdim(x,1,j)))
    end
    values = adapt(
        prob.array_type,
        zeros(prod(length.(grid)), length(prob.itpflags))
    )
    itp = ntuple(size(values)[2]) do j
        _interpolate(
            grid,
            zeros(length.(grid)),
            prob.itpflags[j]
        )
        # _interpolate(
        # scale(
        #     adapt(
        #         prob.array_type,
        #         interpolate(
        #             zeros(length.(grid)),
        #             prob.itpflags[j]
        #         )
        #     ),
        #     StepRangeLen{Float32, Float32, Float32, Int64}(0f0,10f0,length(grid[1]))
        # )
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
