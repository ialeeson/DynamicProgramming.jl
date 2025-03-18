@kwdef @concrete struct ValueFunction <: ValueFunctionType
    f
    obj
    grid
    u0
    v0
    itpflags
    solver = missing
    kwargs = (;)
    inner_solver = missing
    inner_kwargs = Returns(())
end

function init!(prob::ValueFunction, dv, grid, policies, itp, parameters, args)

    len = prod(length.(grid))
    for key in keys(prob.v0)

        _dv = view(dv,1+len*(first(key)-1):len*last(key))
        v0 = prob.v0[key]
        dispatch_multi!(f_kernel!, _dv, v0, grid, policies, itp, parameters,
            args; l=prod(length.(grid)))
        
    end
    
end

function step!(prob::ValueFunction, dv, v, grid, policies, itp, parameters, args)

    _interpolate!(itp, v; sz=length.(grid))
    
    dispatch_multi!(u_kernel!, policies, dv, prob.obj, prob.inner_kwargs, prob.inner_solver, grid, policies, itp, parameters, args; l=prod(length.(grid)))
    
end

function f_kernel!(dv, f, grid, policies, itp, parameters, args, chunk)
    
    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(i -> policies[i][cidx], length(policies)))
        v = f(itp, u0, state, parameters, args...)
        for j in eachindex(v)
            idx = i + (j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
    end
    
end

function u_kernel!(du, dv, f, kwargs, solver, grid, policies, itp, parameters, args, chunk) 

    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(j->policies[j][cidx], length(policies)))
        _kwargs = kwargs(itp, u0, state, parameters, args...)
        (; u, v)=inner_solve(solver; kwargs=_kwargs, p=parameters) do u,p
            f(itp, u, state, p, args...)
        end
        for j in eachindex(v)
            idx = i+(j-1) * prod(length.(grid))
            dv[idx] = v[j]
        end
        for j in eachindex(u)
            du[j][cidx] = u[j]
        end
    end
    
end


function u_kernel!(du::T, dv::T, f, kwargs, solver, grid, policies, itp, parameters, args, chunk) where T 

    C = CartesianIndices(length.(grid))
    @inbounds for i in chunk
        cidx = C[i]
        state = State(ntuple(j->grid[j][cidx[j]], length(cidx)), cidx)
        u0 = SVector(ntuple(j->policies[j][cidx], length(policies)))
        _kwargs = kwargs(itp, u0, state, parameters, args...)
        (; u, v)=inner_solve(solver; kwargs=_kwargs, p=parameters) do u,p
            f(itp, u, state, p, args...)
        end
        for j in eachindex(v)
            dv[j][cidx] = v[j]
        end
        for j in eachindex(u)
            du[j][cidx] = u[j]
        end
    end
    
end
