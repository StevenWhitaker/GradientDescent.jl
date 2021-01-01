function preconditioned_gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Number},
    step_type::AbstractStepType,
    preconditioner,
    func::Union{Nothing,<:Function} = nothing;
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

    isempty(stopping_criteria) && error("at least one stopping criterion must be specified")

    # Set up variables
    x = copy(x0)
    x_prev = copy(x)
    grad = Vector{float(eltype(x0))}(undef, length(x0))
    grad_prev = zeros(float(eltype(x0)), length(x0))
    sqrt_preconditioner = sqrt(preconditioner)
    iter = 0
    time_elapsed = 0
    flag = :NOT_DONE
    if !isnothing(func)
        output = Array{Any}(undef, 1)
        output[1] = func(x, iter, time_elapsed)
    end

    # Run preconditioned gradient descent
    time_start = time()
    while flag === :NOT_DONE

        # Update variables
        compute_gradient!(grad, x)
        step_size = get_step_size(step_type, x, grad, preconditioner, iter)
        x .-= step_size .* (preconditioner * grad)
        iter += 1
        time_elapsed = time() - time_start
        isnothing(func) || push!(output, func(x, iter, time_elapsed))

        # Check stopping criteria
        flag = check_stopping_criteria(stopping_criteria, x, x_prev,
                                       sqrt_preconditioner' * grad,
                                       sqrt_preconditioner' * grad_prev,
                                       iter, time_elapsed)

        # Store previous iteration state
        x_prev .= x
        grad_prev .= grad

    end

    return isnothing(func) ? (x, flag) : (x, flag, output)

end

function preconditioned_gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Real},
    step_type::AbstractStepType,
    preconditioner,
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    func::Union{Nothing,<:Function} = nothing;
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

    all(lower_bounds .<= upper_bounds) || error("lower bounds must be less then or equal to upper bounds")
    all(lower_bounds .<= x0 .<= upper_bounds) || error("x0 must satisfy box constraints")
    isempty(stopping_criteria) && error("at least one stopping criterion must be specified")

    # Set up variables
    x = copy(x0)
    x_prev = copy(x)
    grad = Vector{float(eltype(x0))}(undef, length(x0))
    grad_prev = zeros(float(eltype(x0)), length(x0))
    sqrt_preconditioner = sqrt(preconditioner)
    iter = 0
    time_elapsed = 0
    flag = :NOT_DONE
    if !isnothing(func)
        output = Array{Any}(undef, 1)
        output[1] = func(x, iter, time_elapsed)
    end

    # Run preconditioned gradient descent
    time_start = time()
    while flag === :NOT_DONE

        # Update variables
        compute_gradient!(grad, x)
        step_size = get_step_size(step_type, x, grad, preconditioner, iter)
        x .-= step_size .* (preconditioner * grad)
        iter += 1
        time_elapsed = time() - time_start
        isnothing(func) || push!(output, func(x, iter, time_elapsed))

        # Check stopping criteria
        flag = check_stopping_criteria(stopping_criteria, x, x_prev,
                                       sqrt_preconditioner' * grad,
                                       sqrt_preconditioner' * grad_prev,
                                       iter, time_elapsed)

        # Store previous iteration state
        x_prev .= x
        grad_prev .= grad

    end

    return isnothing(func) ? (x, flag) : (x, flag, output)

end

function preconditioned_gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Number},
    step_type::AbstractStepType,
    preconditioner,
    lower_bounds::AbstractVector{<:Real},
    upper_bounds::AbstractVector{<:Real},
    func::Union{Nothing,<:Function} = nothing;
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

    isempty(stopping_criteria) && error("at least one stopping criterion must be specified")

    # Set up variables
    t = _box_constraints_transform(x0, lower_bounds, upper_bounds)
    t_prev = copy(t)
    grad = Vector{float(eltype(t))}(undef, length(t))
    grad_prev = zeros(float(eltype(t)), length(t))
    sqrt_preconditioner = sqrt(preconditioner)
    iter = 0
    time_elapsed = 0
    flag = :NOT_DONE
    if !isnothing(func)
        output = Array{Any}(undef, 1)
        x = _box_constraints_inverse_transform(t, lower_bounds, upper_bounds)
        output[1] = func(x, iter, time_elapsed)
    end

    # Run preconditioned gradient descent
    time_start = time()
    while flag === :NOT_DONE

        # Update variables
        _compute_gradient_box_constraints!(compute_gradient!, grad, t, lower_bounds, upper_bounds)
        step_size = get_step_size(step_type, t, grad, preconditioner, iter)
        t .-= step_size .* (preconditioner * grad)
        iter += 1
        time_elapsed = time() - time_start
        if !isnothing(func)
            _box_constraints_inverse_transform!(x, t, lower_bounds, upper_bounds)
            push!(output, func(x, iter, time_elapsed))
        end

        # Check stopping criteria
        flag = check_stopping_criteria(stopping_criteria, t, t_prev,
                                       sqrt_preconditioner' * grad,
                                       sqrt_preconditioner' * grad_prev,
                                       iter, time_elapsed)

        # Store previous iteration state
        t_prev .= t
        grad_prev .= grad

    end

    x = _box_constraints_inverse_transform(t, lower_bounds, upper_bounds)
    return isnothing(func) ? (x, flag) : (x, flag, output)

end

function _box_constraints_transform(x, lower_bounds, upper_bounds)

    t = copy(x)

    # Indices with both lower and upper bounds
    i = (-Inf .< lower_bounds) .& (upper_bounds .< Inf)
    half_width = (upper_bounds[i] .- lower_bounds[i]) ./ 2
    t[i] .= tan.((x[i] .- half_width .- lower_bounds[i]) ./ (half_width .* (2 / π)))

    # Indices with just lower bounds
    i = (lower_bounds .> -Inf) .& (upper_bounds .== Inf)
    t[i] .= log.(x[i] .- lower_bounds[i])

    # Indices with just upper bounds
    i = (upper_bounds .< Inf) .& (lower_bounds .== -Inf)
    t[i] .= -log.(-x[i] .+ upper_bounds[i])

    # Indices with equal lower and upper bounds
    # This case is needed to avoid NaNs
    i = lower_bounds .== upper_bounds
    t[i] .= zero(eltype(t))

    return t

end

function _box_constraints_inverse_transform(t, lower_bounds, upper_bounds)

    x = copy(t)

    _box_constraints_inverse_transform!(x, t, lower_bounds, upper_bounds)

    return x

end

function _box_constraints_inverse_transform!(x, t, lower_bounds, upper_bounds)

    # Indices with both lower and upper bounds
    i = (-Inf .< lower_bounds) .& (upper_bounds .< Inf)
    half_width = (upper_bounds[i] .- lower_bounds[i]) ./ 2
    x[i] .= half_width .* (2 / π) .* atan.(t[i]) .+ half_width .+ lower_bounds[i]

    # Indices with just lower bounds
    i = (lower_bounds .> -Inf) .& (upper_bounds .== Inf)
    x[i] .= exp.(t[i]) .+ lower_bounds[i]

    # Indices with just upper bounds
    i = (upper_bounds .< Inf) .& (lower_bounds .== -Inf)
    x[i] .= -exp.(-t[i]) .+ upper_bounds[i]

    return

end

function _compute_gradient_box_constraints!(compute_gradient!, grad, t, lower_bounds, upper_bounds)

    # TODO: ∇f(x) .* d_atan(t)

end
