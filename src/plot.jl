function plot(cache::T; λ::Int=10^2, q=(0.3,0.5,0.8), fig=Figure(), grid) where {T}

    (; policies, values, itp) = cache
    of = fig.content == Any[] ? 0 : size(fig.layout)[2]
    
    for j in 1:length(policies)
        axu = Axis(fig[1,j+of],
            title="policies "*"("*string(keys(grid)[j])*")")
        axy = ntuple(length(itp)) do k
            Axis(fig[1+k,j+of],
                title="values "*"("*string(keys(cache.prob.f)[j])*")")
        end
        finegrid = range(first(grid[j]), last(grid[j]), λ*length(grid[j]))
        for l in q
            idx = ntuple(length(grid)) do k
                searchsortedfirst(grid[k],quantile(grid[k],l))
            end
            u = stack(1:length(grid[j])) do i
                policies[j][idx[begin:j-1]...,i , idx[j+1:end]...]
            end
            lines!(axu, grid[j], u)
            grid = if itp[1] isa WeightedInterpolation
                itp[1].itp.itp.ranges
            elseif itp[1] isa Interpolations.Extrapolation
                itp[1].itp.ranges
            else
                itp[1].ranges
            end
            y = ntuple(length(itp)) do s
                stack(1:length(finegrid)) do i
                    x = ntuple(length(grid)) do k
                        k==j ? finegrid[i] : grid[k][idx[k]]
                    end
                    if itp[s] isa WeightedInterpolation
                        itp[s](x...)
                    else
                        itp[s](x...)
                    end
                end
            end
            lines!.(axy, (finegrid,), y)
        end
    end
    fig
    
end
function plot(cache::EconomicModelCache; λ::Int=10^2)
    fig = Figure()
    for c in cache.cache
        plot(c; λ, fig, grid=cache.prob.grid)
    end
    fig
end
function plot(cache::SteadyStateCache; λ::Int=10^2)
    (; grid, policies, policies, grid) = cache
    f = plot(cache.cache; λ)
    for i in 1:length(policies)
        ax = Axis(f[size(f.layout)[1]+1,1:size(f.layout)[2]],
            title="distribution "*"("*string(keys(grid)[i])*")")
        inds = collect(1:length(grid))
        deleteat!(inds, i)
        barplot!(ax, grid[i], vec(sum(cache.distribution; dims=inds)))
    end
    f
    # Label(
    #     fig[0,of+1:of+length(policies)],
    #     "Steps: $(cache.nsteps), Err: $(round(cache.resid; sigdigits=3))",
    #     tellwidth=false
    # )
end

function plot(cache::ShootingMethodCache; q=(0.1, 0.5, 0.9), T=cache.prob.T, f=())

    (; policies) = cache
    (; grid, grid) = cache.sscache
    (; prob, parameters, grid, values, itp) = cache.cache
    
    fig = Figure()
    for (i,agg) in enumerate(eachslice(cache.aggregates;dims=1))
        ax1 = Axis(fig[1,i],
            title="aggregates "*"("*string(keys(grid)[i])*")")
        lines!(ax1, 1:T, (agg[1:T].-agg[1])./agg[1])
    end
    axu = ntuple(i->Axis(fig[2,i],
        title="policies "*"("*string(keys(grid)[i])*")"),
        length(policies[1]))
    axy = ntuple(i->Axis(fig[3,i],
        title="values "*"("*string(keys(prob.f)[i])*")"),
        length(values))
    axf = ntuple(i->Axis(fig[4,i],
        title="values "*"("*string(keys(f)[i])*")"),
        length(f))
    axf
    for l in q
        idx = ntuple(length(grid)) do k
            searchsortedfirst(grid[k],quantile(grid[k],l))
        end
        u = ntuple(length(axu)) do i
            stack(t->policies[t][i][idx...], 1:T)
        end
        Δu = ntuple(length(axu)) do i
            stack(t->(u[i][t]-u[i][1])/u[i][1], 1:T)
        end
        y = ntuple(length(axy)) do i
            stack(1:T) do t
                s = getindex.(grid, idx)
                _u = ntuple(s -> u[s][t], length(u))
                prob.f[i](fv(_u,idx,itp), _u, s, parameters)
            end
        end
        Δy = ntuple(length(axy)) do i
            stack(t->(y[i][t]-y[i][1])/y[i][1], 1:T)
        end
        vf = ntuple(length(axf)) do i
            stack(1:T) do t
                s = getindex.(grid, idx)
                _u = ntuple(s -> u[s][t], length(u))
                f[i](fv(_u,idx,itp), _u, s, parameters)
            end
        end
        Δvf = ntuple(length(axf)) do i
            stack(t->(vf[i][t]-vf[i][1])/vf[i][1], 1:T)
        end
        lines!.(axu, (1:T,), Δu)
        lines!.(axy, (1:T,), Δy)
        lines!.(axf, (1:T,), Δvf)
    end
    fig
    
end


function plot(cache::ValueFunctionCache;
    λ::Int=10^2, q=(0.3,0.5,0.8), fig=Figure())

    (; policies) = cache
    (; grid) = cache.prob
    
    of = fig.content == Any[] ? 0 : size(fig.layout)[2]
    for j in 1:length(policies)
        axu = Axis(fig[1,j+of],
            title="policies "*"("*string(keys(grid)[j])*")")
        for l in q
            idx = ntuple(length(grid)) do k
                searchsortedfirst(grid[k],quantile(grid[k],l))
            end
            u = stack(1:length(grid[j])) do i
                policies[j][idx[begin:j-1]...,i , idx[j+1:end]...]
            end
            lines!(axu, grid[j], u)
        end
        
        for (k,itp) in enumerate(Iterators.flatten(cache.itp))
            axy = Axis(fig[1+k,j+of],
                title="values "*"("*string(keys(cache.prob.vfs)[k])*")")
            finegrid = range(first(grid[j]), last(grid[j]), λ*length(grid[j]))
            for l in q
                idx = ntuple(length(grid)) do k
                    searchsortedfirst(grid[k],quantile(grid[k],l))
                end
                grid = if itp isa WeightedInterpolation
                    itp.itp.itp.ranges
                elseif itp isa Interpolations.Extrapolation
                    itp.itp.ranges
                else
                    itp.ranges
                end
                y = stack(1:length(finegrid)) do i
                    x = ntuple(length(grid)) do s
                        s==j ? finegrid[i] : grid[s][idx[s]]
                    end
                    itp(x...)
                end
                lines!(axy, finegrid, y)
            end
        end
    end

    fig
end

function plot(cache::InnerValueFunctionCache;
    λ::Int=10^2, q=(0.3,0.5,0.8), fig=Figure())

    (; grid, policies) = cache
        
end
