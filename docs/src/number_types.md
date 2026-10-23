# Number types

A `Float64` state runs in PETSc's double-precision build and a `ComplexF64` state in its
double-precision complex build. A `Float32` or `ComplexF32` state runs in the matching
single-precision build when the span is `Float32` as well, following OrdinaryDiffEq's advice
to give a single-precision problem a `Float32` span, and in the double-precision build when
the span is `Float64`, with the saved states given back in single precision. DiffEqBase
promotes the span to the type of `dt`, so a single-precision solve takes `dt = 0.01f0`
rather than `dt = 0.01`, and it makes a whole-number span `Float64`. Any other real state,
whole numbers included, is solved in `Float64` and comes back in it. Where PETSc.jl has not
loaded the single-precision build, as with a library set with `PETSc.set_library!`, a
single-precision state runs in the double-precision one, and a problem whose build is not
loaded at all is refused with an `ArgumentError` that names it. On 32-bit x86 a
single-precision state always runs in the double-precision build: there PETSc_jll's
single-precision builds end BDF and ARKIMEX solves in failure at stops that its
double-precision build takes.

Times are in the type of the clock PETSc steps on: `Float32` for a single-precision state
with a `Float32` span and `Float64` otherwise. That covers `sol.t` and the integrator's `t`
and `dt`, and the integrator's `u` is in the type PETSc steps. OrdinaryDiffEq gives the
times in the span's type whatever the state; here a `Float64` state with a `Float32` span is
stepped on PETSc's `Float64` clock, and rounding its times to `Float32` would give states
close together the same time.

Single precision has seven digits, which bounds the clock as well as the state. The first
step, and the step after a stop, are at least four ulps of `t`, since PETSc cannot take a
step whose stages have no room between its ends; a single-precision solve is stepped by
this package's integrator, which lands on each stop and on the final time itself, where
PETSc's own landing would refuse a step under about `1e-6` before a final time below 1.
Tolerances finer than single precision can resolve, about `1e-7`, are accepted but buy
nothing past its rounding, and can drive PETSc's adaptive step below what the clock can
tell apart, which ends the solve with a failed retcode. PETSc's single build also takes the
norms its Newton and Krylov iterations stop on in single precision, and a vector whose
entries are all below about `1e-19` in size has a norm of zero there. An implicit solve of
such a state can then stop without moving it and still report success, so this package
warns when a solve starts on such a state, or fails on one; a state that decays there from
above is within any coarser tolerance. Rescale the problem, or give it a `Float64` span.

With a complex state, times, `dt`, `saveat`, `tstops` and the tolerances stay real, and
PETSc's error norms take each component's modulus; a tolerance given as a complex number
with a zero imaginary part is taken as its real part. A `ContinuousCallback`'s condition has
to return a real number, such as `real(u[1]) - 0.5`, since a root is a sign change. The
implicit methods' Newton iteration needs a holomorphic `f`, one that does not go through
`conj`, `abs`, `real` or `imag` of the state. ForwardDiff takes no complex numbers, so
without a `jac` the Jacobian is differentiated along the real parts of the state, which for
a holomorphic `f` is its complex Jacobian, and a sparse `jac_prototype`, or the pattern a
sparse backend is given, is coloured as for a real state. A check at the start compares the
derivatives along the real and the imaginary parts and refuses an `f` that is not
holomorphic; it is best effort, and can miss a term too small to show near the initial
state. `AutoFiniteDiff()` and a hand-written `jac` are not
checked. The explicit methods take any `f`.
