@testset "Operators" begin
  P5(r::AbstractVector{T}) where T = T(1)/T(8) * (T(15) * r - T(70) * r.^3 +
                                                  T(63) * r.^5)

  P6(r::AbstractVector{T}) where T =
    T(1)/T(16) * (-T(5) .+ T(105) * r.^2 - T(315) * r.^4 + T(231) * r.^6)
  DP6(r::AbstractVector{T}) where T =
    T(1)/T(16) * (T(2 * 105) * r - T(4 * 315) * r.^3 + T(6 * 231) * r.^5)

  IPN(::Type{T}, N) where T = T(2)/ T(2 * N + 1)

  N = 6
  for test_type ∈ (Float32, Float64, BigFloat)
    r, w = Canary.lglpoints(test_type, N)
    D = Canary.spectralderivative(r)
    x = LinRange{test_type}(-1, 1, 101)
    I = Canary.interpolationmatrix(r, x)

    @test sum(P5(r).^2 .* w) ≈ IPN(test_type, 5)
    @test D * P6(r) ≈ DP6(r)
    @test I * P6(r) ≈ P6(x)
  end

  for test_type ∈ (Float32, Float64, BigFloat)
    r, w = Canary.lgpoints(test_type, N)
    D = Canary.spectralderivative(r)

    @test sum(P5(r).^2 .* w) ≈ IPN(test_type, 5)
    @test sum(P6(r).^2 .* w) ≈ IPN(test_type, 6)
    @test D * P6(r) ≈ DP6(r)
  end
end
