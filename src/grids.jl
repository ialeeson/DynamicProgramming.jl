abstract type Grid end

struct Markov{I,F} <: Grid
    grid::I
    weights::Matrix{F}
end
