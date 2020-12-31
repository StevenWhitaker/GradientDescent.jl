function test_gradient_descent_1()

    A = Diagonal([1, 4, 3, 3])
    cost_function = x -> norm(A * x)^2

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
    cost_function = x -> norm(A * x)^2

    compute_gradient! = (grad, x) -> grad .= A' * (A * x)
    x0 = ones(size(A, 2))
    step_type = BacktrackingLineSearch(cost_function)

    (x, flag) = gradient_descent(compute_gradient!, x0, step_type)
    return norm(x) <= 1e-14 &&
           flag === :GRAD_TOL_ABS_LIMIT

end

@testset "Gradient Descent" begin

    @test test_gradient_descent_1()
    @test test_gradient_descent_2()

end
