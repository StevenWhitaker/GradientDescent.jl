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

@testset "Preconditioned Gradient Descent" begin

    @test test_preconditioned_gradient_descent_1()
    @test test_preconditioned_gradient_descent_2()

end
