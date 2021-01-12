function test_gradient_descent_1()

    A = Diagonal([1, 4, 3, 3])
    # cost_function = x -> 0.5 * norm(A * x)^2 # Not needed

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = FixedStepSize(1 / opnorm(A)^2)
    func = (x, iter, t) -> nothing

    (x, flag, output) = gradient_descent(compute_gradient!, x0, step_type, func)
    return norm(x) <= 1e-12 &&
           flag === :GRAD_TOL_ABS_LIMIT &&
           all(output .=== nothing)

end

function test_gradient_descent_2()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> 0.5 * norm(A * x)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type)
    return norm(x) <= 1e-12 &&
           flag === :GRAD_TOL_ABS_LIMIT

end

function test_gradient_descent_3()

    A = Diagonal([1, 4, 3, 3])
    # cost_function = x -> 0.5 * norm(A * x)^2 # Not needed

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = FixedStepSize(1 / opnorm(A)^2)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = GradientTolerance(1e-10)

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                                 upper_bounds; stopping_criteria)
    return x ≈ [0, 0, 0, 1] &&
           flag === :GRAD_TOL_LIMIT

end

function test_gradient_descent_4()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> 0.5 * norm(A * x)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = XTolerance(1e-9)

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                                 upper_bounds; stopping_criteria)
    return isapprox(x, [0, 0, 0, 1]; atol = 1e-7) &&
           flag === :X_TOL_LIMIT

end

function test_gradient_descent_5()

    A = Diagonal([1, 4, 3, 3])
    y = ones(size(A, 1))
    cost_function = x -> 0.5 * norm(A * x - y)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x - y)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria = XTolerance(1e-9)

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                                 upper_bounds; stopping_criteria)
    xhat = A \ y
    xhat[4] = lower_bounds[4]
    return isapprox(x, xhat; atol = 1e-8) &&
           flag === :X_TOL_LIMIT

end

function test_gradient_descent_6()

    A = Diagonal([1, 4, 3, 1])
    y = ones(size(A, 1))
    cost_function = x -> 0.5 * norm(A * x - y)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x - y)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                                 upper_bounds)
    xhat = A \ y
    xhat[4] = lower_bounds[4]
    return x ≈ xhat &&
           flag === :GRAD_TOL_ABS_LIMIT

end

function test_gradient_descent_7()

    A = Diagonal([1, 4, 3, 3])
    # cost_function = x -> 0.5 * norm(A * x)^2 # Not needed

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    niter = 10
    step_type = ComputedStepSize(
        (x, grad, P, iter) -> 0 <= iter <= niter ? 1 / opnorm(A)^2 : 0
    )
    lower_bounds = [-10, -5, -Inf, 1]
    upper_bounds = [10, Inf, 5, 1]
    stopping_criteria_1 = MaxIterations(niter)
    stopping_criteria_2 = MaxIterations(niter + 10)

    (x_1,) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                              upper_bounds;
                              stopping_criteria = stopping_criteria_1)
    (x_2,) = gradient_descent(compute_gradient!, x0, step_type, lower_bounds,
                              upper_bounds;
                              stopping_criteria = stopping_criteria_2)
    return x_1 == x_2

end

@testset "Gradient Descent" begin

    @test test_gradient_descent_1()
    @test test_gradient_descent_2()
    @test test_gradient_descent_3()
    @test test_gradient_descent_4()
    @test test_gradient_descent_5()
    @test test_gradient_descent_6()
    @test test_gradient_descent_7()

end
