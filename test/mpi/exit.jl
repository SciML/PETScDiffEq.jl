using MPI, PETScDiffEq, SciMLBase

MPI.Init()
const comm = MPI.COMM_WORLD
const n = MPI.Comm_rank(comm) + 1
const prob = SciMLBase.ODEProblem((du, u, p, t) -> (du .= -u; nothing), ones(n), (0.0, 1.0))
const live = [PETScDiffEq._setup(prob, TSRK("5dp"; comm); dt = 0.1) for _ in 1:8]
const integ = SciMLBase.init(prob, TSRK("5dp"; comm); dt = 0.1)
SciMLBase.step!(integ)
