abstract type AbstractExample end

list_examples() = subtypes(AbstractExample)

include("models/item-response-model.jl")
include("models/satellite-ssm.jl")
include("models/ar1.jl")

# This model won't work with Enzyme.jl.
include("models/fails/satellite-ssm-matrix.jl")
