function preconditioned_gradient_descent(
    compute_gradient!::Function,
    x0::AbstractVector{<:Number},
    step_type::AbstractStepType,
    preconditioner::AbstractMatrix{<:Number},
    func::Union{Nothing,<:Function} = nothing;
    # TODO: Implement Stopping Criteria
    # TODO: Implement get_default_stopping_criteria
    stopping_criteria::StoppingCriteria = get_default_stopping_criteria()
)

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
