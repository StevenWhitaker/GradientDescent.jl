module GradientDescent

using LinearAlgebra: norm, I

export BacktrackingLineSearch
export FixedStepSize

export GradientTolerance
export GradientToleranceAbsolute
export GradientToleranceRelative
export MaxIterations
export MaxTime
export XTolerance
export XToleranceRelative

export gradient_descent
export preconditioned_gradient_descent

include("StepType.jl")
include("StoppingCriteria.jl")

include("preconditioned_gradient_descent.jl")
include("gradient_descent.jl")

end
