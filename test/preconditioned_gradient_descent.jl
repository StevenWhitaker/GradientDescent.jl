function test_preconditioned_gradient_descent_1()

    A = Diagonal([1, 4, 3, 3])
    # cost_function = x -> 0.5 * norm(A * x)^2 # Not needed

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = FixedStepSize(1)
    preconditioner = inv(A' * A)
    func = (x, iter, t) -> nothing

    (x, flag, output) = preconditioned_gradient_descent(compute_gradient!, x0,
                                                step_type, preconditioner, func)
    return norm(x) == 0 &&
           flag === :GRAD_TOL_ABS_LIMIT &&
           all(output .=== nothing)

end

function test_preconditioned_gradient_descent_2()

    Random.seed!(0)

    A = randn(10, 5)
    cost_function = x -> 0.5 * norm(A * x)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = randn(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    preconditioner = inv(Diagonal(vec(sum(abs.(A' * A), dims = 2))))
    stopping_criteria = MaxIterations(300)

    (x, flag) = preconditioned_gradient_descent(compute_gradient!, x0,
                                   step_type, preconditioner; stopping_criteria)
    return norm(x) <= 1e-12 &&
           flag === :ITERATION_LIMIT

end

function test_preconditioned_gradient_descent_3()

    A = Diagonal([1, 4, 3, 3])
    # cost_function = x -> 0.5 * norm(A * x)^2 # Not needed

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = FixedStepSize(1e-3)
    preconditioner = inv(A' * A)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = GradientTolerance(1e-10)

    (x, flag) = preconditioned_gradient_descent(compute_gradient!, x0,
        step_type, preconditioner, lower_bounds, upper_bounds;
        stopping_criteria)
    return x ≈ [0, 0, 0, 1] &&
           flag === :GRAD_TOL_LIMIT

end

function test_preconditioned_gradient_descent_4()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> 0.5 * norm(A * x)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    preconditioner = inv(A' * A)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = XTolerance(1e-8)

    (x, flag) = preconditioned_gradient_descent(compute_gradient!, x0,
        step_type, preconditioner, lower_bounds, upper_bounds;
        stopping_criteria)
    return isapprox(x, [0, 0, 0, 1]; atol = 1e-8) &&
           flag === :X_TOL_LIMIT

end

function test_preconditioned_gradient_descent_5()

    A = Diagonal([1, 4, 3, 3])
    y = ones(size(A, 1))
    cost_function = x -> 0.5 * norm(A * x - y)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x - y)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    preconditioner = inv(A' * A)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = XTolerance(1e-9)

    (x, flag) = preconditioned_gradient_descent(compute_gradient!, x0,
        step_type, preconditioner, lower_bounds, upper_bounds;
        stopping_criteria)
    xhat = A \ y
    xhat[4] = lower_bounds[4]
    return isapprox(x, xhat; atol = 1e-8) &&
           flag === :X_TOL_LIMIT

end

function test_preconditioned_gradient_descent_6()

    A = Diagonal([1, 4, 3, 1])
    y = ones(size(A, 1))
    cost_function = x -> 0.5 * norm(A * x - y)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x - y)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    preconditioner = inv(A' * A)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]

    (x, flag) = preconditioned_gradient_descent(compute_gradient!, x0,
        step_type, preconditioner, lower_bounds, upper_bounds)
    xhat = A \ y
    xhat[4] = lower_bounds[4]
    return x ≈ xhat &&
           flag === :GRAD_TOL_ABS_LIMIT

end

function test_box_constraints_transform_1()

    Random.seed!(0)

    N = 2000
    T = div(N, 5)
    t = [randn(4T); zeros(T)]
    lower_bounds = [randn(2T); fill(-Inf, 2T); randn(T)]
    upper_bounds = [randn(T); fill(Inf, T); randn(T); fill(Inf, T); lower_bounds[4T+1:end]]
    for i = 1:N
        if upper_bounds[i] < lower_bounds[i]
            tmp = lower_bounds[i]
            lower_bounds[i] = upper_bounds[i]
            upper_bounds[i] = tmp
        end
    end

    x = GradientDescent._box_constraints_inverse_transform(t, lower_bounds, upper_bounds)
    t2 = GradientDescent._box_constraints_transform(x, lower_bounds, upper_bounds)

    return t ≈ t2 && all(lower_bounds .<= x .<= upper_bounds)

end

function test_compute_gradient_box_constraints_1()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> 0.5 * norm(A * x)^2
    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    grad = zeros(size(A, 2))
    x = ones(size(A, 2))
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    t = GradientDescent._box_constraints_transform(x, lower_bounds, upper_bounds)
    cost_function_box_constraints = t -> cost_function(
        GradientDescent._box_constraints_inverse_transform(t, lower_bounds,
                                                           upper_bounds))

    GradientDescent._compute_gradient_box_constraints!(compute_gradient!, grad,
                                               x, t, lower_bounds, upper_bounds)

    ϵ = 1e-5
    grad_t = zeros(size(grad))
    for i = 1:length(grad_t)
        ϵvec = zeros(length(grad_t))
        ϵvec[i] = ϵ
        grad_t[i] = (cost_function_box_constraints(t + ϵvec) -
                     cost_function_box_constraints(t - ϵvec)) / 2ϵ
    end

    return grad ≈ grad_t

end

@testset "Preconditioned Gradient Descent" begin

    @test test_preconditioned_gradient_descent_1()
    @test test_preconditioned_gradient_descent_2()
    @test test_preconditioned_gradient_descent_3()
    @test test_preconditioned_gradient_descent_4()
    @test test_preconditioned_gradient_descent_5()
    @test test_preconditioned_gradient_descent_6()

end

@testset "Box Constraints Transform" begin

    @test test_box_constraints_transform_1()
    @test test_compute_gradient_box_constraints_1()

end
