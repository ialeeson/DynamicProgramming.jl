@kwdef @concrete struct Howard
    f
    solver
    maxiters = 10^2
    abstol = 1e-10
end

function _step!(dv, prob::Howard, grid, policies, residuals, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = ntuple(j->policies[j][cidx], length(policies))
        v = prob.f(itp, u0, state, parameters, args...)
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
    end
    
end
