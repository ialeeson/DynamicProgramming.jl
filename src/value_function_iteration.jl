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
