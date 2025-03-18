@kwdef @concrete struct NaiveEulerEquation <: ValueFunctionType
    f
    grid
    ee
    u0
    v0
    itpflags
    solver = missing
    kwargs = (; maxiters=10^2, abstol=1e-20)
    inner_solver = missing
    inner_kwargs = Returns(())
end

function _step!(dv, prob::NaiveEulerEquation, grid, policies, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(i -> policies[i][cidx], length(policies)))
        kwargs = prob.inner_kwargs(u0, state, parameters, args...)
        u = inner_solve(prob.inner_solver; kwargs, p=parameters) do u,p
            prob.ee(itp, u, state, p, args...)
        end
        for key in keys(prob.f)
            f = prob.f[key]
            v = f(itp, u, state, parameters, args...)
            for j in eachindex(v)
                idx = i + (first(key)-1+j-1) * prod(length.(grid))
                dv[idx] = v[j]
            end
        end
        for j in eachindex(u)
            policies[j][cidx] = u[j]
        end
    end
    
end
