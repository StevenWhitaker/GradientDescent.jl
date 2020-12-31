using GradientDescent
using LinearAlgebra
using Random
using Test

@testset "GradientDescent.jl" begin
    include("gradient_descent.jl")
    include("preconditioned_gradient_descent.jl")
end
