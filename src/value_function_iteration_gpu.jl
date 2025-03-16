# @kernel function _init!(dv, prob::Union{ValueFunction,EulerEquation}, grid, policies, residuals, itp, parameters, args, chunk)

#     C = CartesianIndices(length.(grid))
#     i = @index(Global)
#     cidx = C[i]
#     state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
#     u0 = ntuple(j->policies[j][cidx], length(policies))
#     v = prob.v0(u0, state, parameters, args...)
#     for j in eachindex(v)
#         idx = i+(j-1) * prod(length.(grid))
#         dv[idx] = v[j]
#     end
    
# end
