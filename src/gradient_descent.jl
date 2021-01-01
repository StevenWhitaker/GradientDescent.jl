function gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Number},
    step_type::AbstractStepType,
    func::Union{Nothing,<:Function} = nothing;
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

    return preconditioned_gradient_descent(compute_gradient!, x0, step_type, I,
                                           func; stopping_criteria)

end

function gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Real},
    step_type::AbstractStepType,
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    func::Union{Nothing,<:Function} = nothing;
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

    return preconditioned_gradient_descent(compute_gradient!, x0, step_type, I,
                                           lower_bounds, upper_bounds, func;
                                           stopping_criteria)

end
