function test_preconditioned_gradient_descent_1()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> norm(A * x)^2

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
    cost_function = x -> norm(A * x)^2

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

@testset "Preconditioned Gradient Descent" begin

    @test test_preconditioned_gradient_descent_1()
    @test test_preconditioned_gradient_descent_2()

end

@testset "Box Constraints Transform" begin

    @test test_box_constraints_transform_1()

end
