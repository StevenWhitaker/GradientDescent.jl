abstract type AbstractStoppingCriterion end

StoppingCriteria = Union{<:AbstractStoppingCriterion,<:NTuple{N,AbstractStoppingCriterion} where N}

Base.isempty(::AbstractStoppingCriterion) = false
Base.isempty(::NTuple{N,AbstractStoppingCriterion}) where N = N <= 0

function check_stopping_criteria(
    criteria::NTuple{N,AbstractStoppingCriterion},
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
) where N

    for criterion in criteria
        flag = check_stopping_criteria(criterion, x, x_prev, grad, grad_prev,
                                       iter, time_elapsed)
        flag === :NOT_DONE || return flag
    end
    return :NOT_DONE

end

struct XTolerance <: AbstractStoppingCriterion
    x_tol::Float64
end

function check_stopping_criteria(
    x_tol::XTolerance,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return norm(x - x_prev) <= x_tol.x_tol ? :X_TOL_LIMIT : :NOT_DONE

end

struct XToleranceRelative <: AbstractStoppingCriterion
    x_tol_rel::Float64
end

function check_stopping_criteria(
    x_tol_rel::XToleranceRelative,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return norm(x - x_prev) / norm(x_prev) <= x_tol_rel.x_tol_rel ? :X_TOL_REL_LIMIT : :NOT_DONE

end

struct GradientTolerance <: AbstractStoppingCriterion
    grad_tol::Float64
end

function check_stopping_criteria(
    grad_tol::GradientTolerance,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return norm(grad - grad_prev) <= grad_tol.grad_tol ? :GRAD_TOL_LIMIT : :NOT_DONE

end

struct GradientToleranceRelative <: AbstractStoppingCriterion
    grad_tol_rel::Float64
end

function check_stopping_criteria(
    grad_tol_rel::GradientToleranceRelative,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return norm(grad - grad_prev) / norm(grad_prev) <= grad_tol_rel.grad_tol_rel ? :GRAD_TOL_REL_LIMIT : :NOT_DONE

end

struct GradientToleranceAbsolute <: AbstractStoppingCriterion
    grad_tol_abs::Float64
end

function check_stopping_criteria(
    grad_tol_abs::GradientToleranceAbsolute,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return norm(grad) <= grad_tol_abs.grad_tol_abs ? :GRAD_TOL_ABS_LIMIT : :NOT_DONE

end

struct MaxIterations <: AbstractStoppingCriterion
    max_iterations::Int
end

function check_stopping_criteria(
    max_iterations::MaxIterations,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return iter >= max_iterations.max_iterations ? :ITERATION_LIMIT : :NOT_DONE

end

struct MaxTime <: AbstractStoppingCriterion
    max_time::Float64
end

function check_stopping_criteria(
    max_time::MaxTime,
    x::AbstractVector{<:Number},
    x_prev::AbstractVector{<:Number},
    grad::AbstractVector{<:Number},
    grad_prev::AbstractVector{<:Number},
    iter::Integer,
    time_elapsed::Real
)

    return time_elapsed >= max_time.max_time ? :TIME_LIMIT : :NOT_DONE

end

get_default_stopping_criteria() = (GradientToleranceAbsolute(1e-12),
                                   MaxTime(24 * 60 * 60))
