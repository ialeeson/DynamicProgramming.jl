__interpolate(grid, values, flags) =
    extrapolate(scale(interpolate(values, flags), grid), Interpolations.Flat())

_interpolate(grid, values, flags) = __interpolate(grid, values, flags)
function _interpolate(grid, values, flags::Tuple)
    grid = ntuple(length(flags)) do i
        if flags[i] isa NoInterp
            range(1,length(grid[i]),length(grid[i]))
        else
            grid[i]
        end
    end
    __interpolate(grid, values, flags)
end
_interpolate(grid, values, flags::Gridded) =
    extrapolate(interpolate(grid, values, flags), Linear())

_interpolate!(itp, A, sz) = _interpolate!(
    Interpolations.itpflag(itp),
    Interpolations.coefficients(itp),
    A,
    sz
)
_interpolate!(itp, coefs, A, sz) = copyto!(coefs,A)
_interpolate!(itp::Interpolations.Extrapolation, A, sz) =
    _interpolate!(itp.itp, A, sz)
function _interpolate!(flags, coefs::T, A::R, sz) where {T<:Interpolations.OffsetArray,R<:AbstractArray}

    for (i,cidx) in enumerate(CartesianIndices(sz))
        coefs[cidx] = A[i]
    end
    for j in 1:length(size(coefs))
        for i in minimum(axes(coefs,j)):0
            selectdim(coefs,j,i) .= 0.0
        end
        for i in sz[j]+1:maximum(axes(coefs,j))
            selectdim(coefs,j,i) .= 0.0
        end
    end
    Interpolations.prefilter!(eltype(coefs), coefs, flags)
    
end

### Weighted
struct Weighted
    pre
    post
    weights
    itpflags
    Weighted(pre, post, weights, flags...) = begin
        for v in values(weights)
            !(all(sum(v; dims=2) .≈ 1)) &&
            error("Weight matrix is not valid.")
        end
        new(pre, post, weights, flags)
    end
end
@kwdef @concrete struct WeightedInterpolation
    pre
    post
    itp
    weights
    dims
    tmp0
    tmp1
end
@inline (i::WeightedInterpolation)(x...) = i.itp(x...)

function _interpolate(grid, _values, flags::Weighted)
    pre = flags.pre
    post = flags.post
    itp = _interpolate(grid, _values, flags.itpflags)
    dims = (keys(flags.weights)...,)
    tmp0 = itensor(
        similar(_values),
        Index.(length.(grid))
    )
    tmp1 = itensor(
        similar(_values),
        ntuple(length(grid)) do i
            i in dims ? Index(length(grid[i])) : inds(tmp0)[i]
        end
    )
    weights = Tuple(
        itensor(flags.weights[k], inds(tmp1)[k], inds(tmp0)[k])
        for k in keys(flags.weights)
    )
    WeightedInterpolation(; pre, post, itp, weights, dims, tmp0, tmp1)
end
function _interpolate!(witp::WeightedInterpolation, A, sz)
    (; pre, post, itp, tmp0, tmp1, weights) = witp
    map!(pre, ITensors.tensor(tmp0), A)
    tmp1 .= post.(*(tmp0, weights...))
    _interpolate!(itp, ITensors.tensor(tmp1), sz)
end

### Integral
@kwdef struct Integral
    weights
    nodes = 1:size(weights)[1]
    itpflags
end

struct IntegralInterpolation{W,N,IT}
    nodes::N
    weights::W
    itp::IT
end
(t::IntegralInterpolation)(f,j) = sum(
    t.itp(f(i,x)...)*t.weights[i] for (i,x) in enumerate(t.nodes)
)
(t::IntegralInterpolation{Matrix{F}})(f,j) where {F} = sum(
    t.itp(f(i,x)...)*t.weights[j,i] for (i,x) in enumerate(t.nodes)
)
function _interpolate(grid, values, int::Integral)
    (; nodes, weights, itpflags) = int
    itp = _interpolate(grid, values, itpflags)
    IntegralInterpolation(nodes, weights, itp)
end
_interpolate!(itp::IntegralInterpolation, A, sz) = _interpolate!(itp.itp, A, sz)
