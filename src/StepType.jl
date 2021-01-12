abstract type AbstractStepType end

struct FixedStepSize{T<:Real} <: AbstractStepType
    step_size::T
end

get_step_size(fixed_step::FixedStepSize, args...) = fixed_step.step_size

struct ComputedStepSize{F<:Function} <: AbstractStepType
    func::F
end

get_step_size(computed::ComputedStepSize, args...) = computed.func(args...)

struct BacktrackingLineSearch{F<:Function,T<:Real} <: AbstractStepType
    cost_function::F
    slope::T
    shrinkage::T
    max_step_size::T

    function BacktrackingLineSearch(
        cost_function::F;
        slope::Real = 0.5,
        shrinkage::Real = 0.5,
        max_step_size::Real = 1
    ) where {F<:Function}

        0 < slope < 1 || error("Required: 0 < slope < 1")
        0 < shrinkage < 1 || error("Required: 0 < shrinkage < 1")
        max_step_size > 0 || error("Required: max_step_size > 0")
   
        args = promote(slope, shrinkage, max_step_size)
        T = eltype(args)
        return new{F,T}(cost_function, args...)
    
    end
end

function get_step_size(
    bls::BacktrackingLineSearch,
    x::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    iter::Integer
)

    cost = bls.cost_function(x)
    step_size = bls.max_step_size
    while bls.cost_function(x .- step_size .* grad) > (cost - bls.slope * step_size * (grad' * grad))
        step_size *= bls.shrinkage
    end
    return step_size

end

function get_step_size(
    bls::BacktrackingLineSearch,
    x::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    preconditioner,
    iter::Integer
)

    cost = bls.cost_function(x)
    P_grad = preconditioner * grad
    step_size = bls.max_step_size
    while bls.cost_function(x .- step_size .* P_grad) > (cost - bls.slope * step_size * (grad' * P_grad))
        step_size *= bls.shrinkage
    end
    return step_size

end
