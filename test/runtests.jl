using Vardr
using Test

include("fixtures.jl")

@testset "Vardr.jl" begin
    @testset "Model" include("test_model.jl")
    @testset "Neural and agent models" include("test_models.jl")
    @testset "Assimilation" include("test_assimilation.jl")
    @testset "What-if" include("test_whatif.jl")
    @testset "Calibration" include("test_calibration.jl")
    @testset "Persistence" include("test_persistence.jl")
    @testset "Streaming" include("test_streaming.jl")
    @testset "ModelingToolkit" include("test_mtk.jl")
    @testset "Validation" include("test_validation.jl")
    @testset "Code quality" include("test_quality.jl")
end
