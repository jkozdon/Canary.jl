using Test
using MPI
using Canary

function main()
  MPI.Init()
  comm = MPI.COMM_WORLD
  crank = MPI.Comm_rank(comm)
  csize = MPI.Comm_size(comm)

  @assert csize == 2

  DFloat = Float64
  Nx = 3
  Ny = 2
  x = range(DFloat(0); length=Nx+1, stop=1)
  y = range(DFloat(0); length=Ny+1, stop=1)
  brick = brickmesh((x,y), (true, true), boundary=[1 3; 2 4],
                    part=crank+1, numparts=csize)
  mesh = partition(comm, brick...)
  mesh = connectmesh(comm, mesh...)

  (elems,
   realelems,
   ghostelems,
   sendelems,
   elemtocoord,
   elemtoelem,
   elemtoface,
   elemtoordr,
   elemtobndy,
   nabrtorank,
   nabrtorecv,
   nabrtosend) = mesh

  globalelemtoface = [2  2  2  2  2  2
                      1  1  1  1  1  1
                      4  4  4  4  4  4
                      3  3  3  3  3  3]

  globalelemtoordr = ones(Int, size(globalelemtoface))
  globalelemtobndy = zeros(Int, size(globalelemtoface))

  if crank == 0
    nrealelem = 3
    globalelems = [1,2,3,4,5,6]
    elemtoelem_expect = [6 4 2 4 5 6
                         5 3 4 4 5 6
                         2 1 5 4 5 6
                         2 1 5 4 5 6]
    nabrtorank_expect = [1]
    nabrtorecv_expect = UnitRange{Int}[1:3]
    nabrtosend_expect = UnitRange{Int}[1:3]
  elseif crank == 1
    nrealelem = 3
    globalelems = [4,5,6,1,2,3]
    elemtoelem_expect = [6 4 2 4 5 6
                         5 3 4 4 5 6
                         3 6 1 4 5 6
                         3 6 1 4 5 6]
    nabrtorank_expect = [0]
    nabrtorecv_expect = UnitRange{Int}[1:3]
    nabrtosend_expect = UnitRange{Int}[1:3]
  end

  @test elems == 1:length(globalelems)
  @test realelems == 1:nrealelem
  @test ghostelems == nrealelem+1:length(globalelems)
  @test elemtoface[:,realelems] == globalelemtoface[:,globalelems[realelems]]
  @test elemtoelem == elemtoelem_expect
  @test elemtobndy == globalelemtobndy[:,globalelems]
  @test elemtoordr == ones(eltype(elemtoordr), size(elemtoordr))
  @test nabrtorank == nabrtorank_expect
  @test nabrtorecv == nabrtorecv_expect
  @test nabrtosend == nabrtosend_expect

  MPI.Finalize()
end

main()
