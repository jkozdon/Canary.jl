using LinearAlgebra
import Base.*
using SparseArrays
using SIMD
using BenchmarkTools


struct TensorProductOperator{dim, Tξ, Tη, Tζ}
  Aξ::Tξ
  Aη::Tη
  Aζ::Tζ
end
TensorProductOperator(Aξ::Tξ, Aη::Tη, Aζ::Tζ) where {Tξ, Tη, Tζ} = TensorProductOperator{3, Tξ, Tη, Tζ}(Aξ, Aη, Aζ)
*(A::TensorProductOperator, B) = TPmul!(zero(B), A, B)

# ξ dimension operators
function TPmul_v1!(C, A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                         UniformScaling{Bool}}, B) where T
  Nq = size(A.Aξ, 1)
  TPmul_v1!(Val(Nq), C, A, B)
end
function TPmul_v1!(::Val{Nq}, C,
              A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                       UniformScaling{Bool}},
                             B) where {Nq, T}

  Nq2 = Nq * Nq
  Nq3 = Nq * Nq * Nq

  @assert length(C) == length(B)
  nelem = div(length(B), Nq3)

  @inbounds for e = 0:nelem-1
    for k = 0:Nq-1, j = 0:Nq-1, i = 0:Nq-1, n = 0:Nq-1
      C[1 + i + j * Nq + k * Nq2 + e * Nq3] +=
         A.Aξ[1 + i + n * Nq] * B[1 + n + j * Nq + k * Nq2 + e * Nq3]
    end
  end
  C
end

function TPmul_v2!(C, A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                         UniformScaling{Bool}}, B) where T
  Nq = size(A.Aξ, 1)
  TPmul_v2!(Val(Nq), C, A, B)
end
function TPmul_v2!(::Val{Nq}, C,
              A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                       UniformScaling{Bool}},
              B,) where {Nq, T}

  Nq2 = Nq * Nq
  Nq3 = Nq * Nq * Nq

  @assert length(C) == length(B)
  nelem = div(length(B), Nq3)

  @inbounds for e = 0:nelem-1
    for k = 0:Nq-1, j = 0:Nq-1, i = 0:Nq-1
      t = C[1 + i + j * Nq + k * Nq2 + e * Nq3]
      for n = 0:Nq-1
        t += A.Aξ[1 + i + n * Nq] * B[1 + n + j * Nq + k * Nq2 + e * Nq3]
      end
      C[1 + i + j * Nq + k * Nq2 + e * Nq3] = t
    end
  end
  C
end

function TPmul_v3!(C, A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                         UniformScaling{Bool}}, B) where T
  Nq = size(A.Aξ, 1)
  TPmul_v3!(Val(Nq), Val(4), C, A, B)
end
function TPmul_v3!(::Val{Nq}, ::Val{VL}, C,
              A::TensorProductOperator{3, Matrix{T}, UniformScaling{Bool},
                                       UniformScaling{Bool}},
              B,) where {Nq, T, VL}
  Nq2 = Nq * Nq
  Nq3 = Nq * Nq * Nq

  @assert length(C) == length(B)
  nelem = div(length(B), Nq3)

  Nv = VL * (div(Nq, VL)-1)
  @inbounds for e = 0:nelem-1
    for k = 0:Nq-1, j = 0:Nq-1
      pCijke = pointer(C,1 + j * Nq + k * Nq2 + e * Nq3)
      for i = 0:VL:Nv
        Cv = vload(Vec{VL, T}, pCijke)
        pAin = pointer(A.Aξ, i+1)
        for n = 0:Nq-1
          Av = vload(Vec{VL, T}, pAin)
          b = B[1 + n + j * Nq + k * Nq2 + e * Nq3]
          Cv = fma(Av, b, Cv)
          pAin += Nq * sizeof(T)
        end
        vstore(Cv, pCijke)
        pCijke += VL * sizeof(T)
      end
      for i = (Nv+VL):Nq-1
        t = C[1 + i + j * Nq + k * Nq2 + e * Nq3]
        for n = 0:Nq-1
          t += A.Aξ[1 + i + n * Nq] * B[1 + n + j * Nq + k * Nq2 + e * Nq3]
        end
        C[1 + i + j * Nq + k * Nq2 + e * Nq3] = t
      end
    end
  end
  C
end


# η dimension operators
function TPmul!(C, A::TensorProductOperator{3, UniformScaling{Bool},
                                    UniformScaling{Bool}, Matrix{T}}, B) where T
  Nq = size(A.Aζ, 1)
  TPmul!(Val(Nq), C, A, B)
end
function TPmul!(::Val{Nq}, C,
              A::TensorProductOperator{3, UniformScaling{Bool},
                                       UniformScaling{Bool}, Matrix{T}},
                             B) where {Nq, T}

  Nq2 = Nq * Nq
  Nq3 = Nq * Nq * Nq

  @assert length(C) == length(B)
  nelem = div(length(B), Nq3)

  @inbounds for e = 0:nelem-1
    for k = 0:Nq-1, j = 0:Nq-1, i = 0:Nq-1, n = 0:Nq-1
      C[1 + i + j * Nq + k * Nq2 + e * Nq3] +=
         A.Aζ[1 + k + n * Nq] * B[1 + i + j * Nq + n * Nq2 + e * Nq3]
    end
  end
  C
end


# η dimension operators
function TPmul!(C, A::TensorProductOperator{3, UniformScaling{Bool}, Matrix{T},
                                    UniformScaling{Bool}}, B) where T
  Nq = size(A.Aη, 1)
  TPmul!(Val(Nq), C, A, B)
end
function TPmul!(::Val{Nq}, C,
              A::TensorProductOperator{3, UniformScaling{Bool},
                                       Matrix{T}, UniformScaling{Bool}},
                             B) where {Nq, T}

  Nq2 = Nq * Nq
  Nq3 = Nq * Nq * Nq

  @assert length(C) == length(B)
  nelem = div(length(B), Nq3)

  @inbounds for e = 0:nelem-1
    for k = 0:Nq-1, j = 0:Nq-1, i = 0:Nq-1, n = 0:Nq-1
      C[1 + i + j * Nq + k * Nq2 + e * Nq3] +=
         A.Aη[1 + j + n * Nq] * B[1 + i + n * Nq + k * Nq2 + e * Nq3]
    end
  end
  C
end

let
  N = 4
  Nq = N + 1
  nelem = 100000

  Typ = Float64

  D = rand(Typ, Nq, Nq)
  Dξ = TensorProductOperator(D, I, I)
  Dη = TensorProductOperator(I, D, I)
  Dζ = TensorProductOperator(I, I, D)

  Q = rand(Typ, Nq^3, nelem)
  dQ = similar(Q)

  Dr = kron(Matrix(I, Nq, Nq), Matrix(I, Nq, Nq), D)
  dQ .= 0
  mul!(dQ, Dr,  Q)
  dQT = copy(dQ)

  dQ .= 0
  TPmul_v1!(dQ, Dξ,  Q)
  println("v1 - matmul")
  @show maximum(abs.(dQ - dQT))

  dQ .= 0
  TPmul_v2!(dQ, Dξ,  Q)
  println("v2 - matmul")
  @show maximum(abs.(dQ - dQT))

  dQ .= 0
  TPmul_v3!(dQ, Dξ,  Q)
  println("v3 - matmul")
  @show maximum(abs.(dQ - dQT))

  println("matmul")
  @btime mul!($dQ, $Dr, $Q)

  println("v1")
  @btime TPmul_v1!($dQ, $Dξ, $Q)

  println("v2")
  @btime TPmul_v2!($dQ, $Dξ, $Q)

  println("v3")
  @btime TPmul_v3!($dQ, $Dξ, $Q)

  #=
  Dr = kron(D, sparse(I, Nq, Nq), sparse(I, Nq, Nq))
  println(7)
  @time dQ .= Dr * Q
  @time dQ .= Dr * Q


  dQ = Dη * Q
  for e = 1:nelem, k = 1:Nq, i = 1:Nq
    Dqtest[i, :, k, e] = D * Q[i, :, k, e]
  end
  @show maximum(abs.(dQ - Dqtest))

  dQ = Dζ * Q
  for e = 1:nelem, j = 1:Nq, i = 1:Nq
    Dqtest[i, j, :, e] = D * Q[i, j, :, e]
  end
  @show maximum(abs.(dQ - Dqtest))
  =#
  nothing
end
