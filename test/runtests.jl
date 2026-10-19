using Vardr
using Test
using Aqua
using JET

@testset "Vardr.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Vardr)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Vardr; target_defined_modules = true)
    end
    # Write your tests here.
end
