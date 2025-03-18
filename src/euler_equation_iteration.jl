@kwdef @concrete struct EulerEquation <: ValueFunctionType
    grid
    f
    ee
    u0
    v0
    itpflags
    solver = missing
    kwargs = (; maxiters=10^2, abstol=1e-20)
    inner_solver = missing
    inner_kwargs = Returns(())
end


function init!(prob::Union{EulerEquation, NaiveEulerEquation}, dv, grid, policies, residuals, itp, parameters, args)

    l = prod(length.(grid))

    for key in keys(prob.v0)

        v0 = prob.v0[key]
        _dv = view(dv,1+l*(first(key)-1):l*last(key))
        _itp = itp[first(key):last(key)]
        step!(v0, _dv, grid, policies, itp, parameters, args)

    end

    _interpolate!(itp, dv; sz=length.(grid))
    
end

function step!(prob::Union{EulerEquation, NaiveEulerEquation}, dv, v, grid, policies, residuals, itp, parameters, args)

    l = prod(length.(grid))
    n = length(prob.f)
    for key in keys(prob.ee)

        ee = prob.ee[key]
        du = policies[first(key):last(key)]
        λ = residuals[first(key):last(key)]
        
        dispatch_multi!(u_kernel!, du, λ, ee, prob.inner_kwargs,
            prob.inner_solver, grid, policies, itp, parameters, args; l)
        
    end

    for key in keys(prob.f)

        f = prob.f[key]
        @views _dv = dv[1+l*(first(key)-1):l*last(key)]
        _itp = itp[first(key):last(key)]
        step!(f, _dv, grid, policies, itp, parameters, args)
        _interpolate!(_itp, _dv; sz=length.(grid))
        
    end
    
end
