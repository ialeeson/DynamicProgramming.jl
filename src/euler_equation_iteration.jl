@kwdef @concrete struct EulerEquation
    f
    grid
    ee
    u0
    v0
    itpflags
    solver = missing
    inner_solver = missing
    policy_prob = missing
    kwargs = Returns(())
    maxiters = 10^2
    abstol = 1e-10
end

function _step!(dv, prob::EulerEquation, grid, policies, residuals, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(i -> policies[i][cidx], length(policies)))
        kwargs = prob.kwargs(u0, state, parameters, args...)
        (; u, r) = inner_solve(prob.inner_solver; kwargs, p=parameters) do u,p
            prob.ee(itp, u, state, p, args...)
        end
        v = prob.f(itp, u, state, parameters, args...)
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
        for j in eachindex(u)
            policies[j][cidx] = u[j]
            residuals[j][cidx] = r[j]
        end
    end
    
end
