### Problem
@kwdef @concrete struct HowardPolicyProblem <: ModelProblem
    f
    callback = cache -> nothing
end
init(prob::HowardPolicyProblem, cache) = @set cache.prob = prob

### Solve
function _step!(prob::HowardPolicyProblem, grid, values, policies, itp, chunk)
    resid = 0.0
    for cidx in chunk
        u_prev = getindex.(policies, (cidx,))
        state = State(getindex.(grid, Tuple(cidx)), cidx, itp)
        @inbounds for j in eachindex(prob.f)
            v = prob.f(u_prev, state)
            resid += (v-values[j][cidx])^2
            values[j][cidx] = v
        end
    end
    resid
end
