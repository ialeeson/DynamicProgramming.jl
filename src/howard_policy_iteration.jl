@kwdef @concrete struct HowardEquation <: ValueFunctionType
    f
    solver
    kwargs = (; maxiters=10^2, abstol=1e-10)
end

step!(f::HowardEquation, dv, grid, policies, itp, parameters, args) =
    dispatch_multi!(f_kernel!, dv, f.f, grid, policies, itp,
        parameters, args; l=prod(length.(grid)))

step!(f::F, dv, grid, policies, itp, parameters, args) where {F<:Function} =
    dispatch_multi!(f_kernel!, dv, f, grid, policies, itp,
        parameters, args; l=prod(length.(grid)))


function solve_howard!(v, fs, grid, policies, itp, parameters, args)
    
    len = prod(length.(grid))
        
    for key in keys(fs)

        @views _v = v[1+len*(first(key)-1):len*last(key)]
        _itp = itp[first(key):last(key)]
                            
        if fs[key] isa HowardEquation
            
            f = fs[key]
            
            sol = solve!(_v, f.solver; f.kwargs...) do du,u

                _interpolate!(_itp, u; sz=length.(grid))
                step!(f, du, grid, policies, itp, parameters, args)
                du .-= u
                
            end

            _v .= sol.u
            
        end
            
    end
    
end
