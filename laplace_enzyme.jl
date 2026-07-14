using Ferrite, SparseArrays, ForwardDiff, PureUMFPACK, Enzyme
import EnzymeCore
import EnzymeCore.EnzymeRules as ER

grid = generate_grid(Quadrilateral, (20, 20));

ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

ch = ConstraintHandler(dh);

∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
);

dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

close!(ch)

function assemble_element!(Ke::Matrix, fe::AbstractVector, cellvalues::CellValues, coords::Vector{<:Vec}, A, B)
    n_basefuncs = getnbasefunctions(cellvalues)
    fill!(Ke, 0)
    fill!(fe, 0)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        x = spatial_coordinate(cellvalues, q_point, coords)
        f_qp = A * sin(x[1]) + B * cos(x[2])
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            fe[i] += f_qp * δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues::CellValues, dh::DofHandler, A, B)
    n_basefuncs = getnbasefunctions(cellvalues)
    T = promote_type(typeof(A), typeof(B))
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(T, n_basefuncs)
    K = allocate_matrix(dh)
    f_global = zeros(T, ndofs(dh))
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        assemble_element!(Ke, fe, cellvalues, getcoordinates(cell), A, B)
        assemble!(assembler, celldofs(cell), Ke)
        assemble!(f_global, celldofs(cell), fe)
    end
    return K, f_global
end

# --- Enzyme + PureUMFPACK ---
#
# Confirmed by experiment: Enzyme.autodiff can trace through the Ferrite
# assembly loop (CellIterator/DofHandler/assemble!) and through apply!(K, f,
# ch) without issue, as long as `cellvalues`/`dh`/`ch` are passed as explicit
# `Const` arguments (not closed-over globals) and reverse mode is run with
# `set_runtime_activity`.
#
# However, letting Enzyme trace *through* PureUMFPACK's `splu`/`\` (the
# Gilbert-Peierls LU with dynamic pivoting/sparse growth) reliably segfaults
# deep in LLVM's MemorySSA pass (confirmed with both the full solve and a
# minimal assembly+splu-only reproduction) - the sparse factorization's
# irregular, data-dependent control flow appears to be more than Enzyme's
# current compiler (bundled LLVM 20 on Julia 1.13.0-rc1) can handle.
#
# Workaround: give Enzyme a hand-written adjoint for the linear solve via
# EnzymeRules, so it treats `linsolve` as an opaque primitive instead of
# differentiating through the factorization/triangular-solve internals. K is
# constant (independent of A, B) here, and K is symmetric, so the reverse-mode
# rule is just another solve with the same factorization: f̄ = F \ ū.
linsolve(F, b::AbstractVector) = F \ b

function ER.augmented_primal(
        config, func::EnzymeCore.Const{typeof(linsolve)}, ::Type{<:EnzymeCore.Duplicated},
        F::Union{EnzymeCore.Const, EnzymeCore.Duplicated}, b::EnzymeCore.Duplicated,
    )
    # Enzyme's static activity analysis cannot prove F (the factorization of
    # K) is Const even though K never actually depends on A, B, so it may
    # call this rule with F::Duplicated instead of F::Const. Either way only
    # F.val (the primal factorization) is used - any shadow on F is ignored
    # since K's real derivative w.r.t. A, B is exactly zero.
    u = func.val(F.val, b.val)
    shadow = zero(u)
    return ER.AugmentedReturn(u, shadow, shadow)
end

function ER.reverse(
        config, func::EnzymeCore.Const{typeof(linsolve)}, ::Type{<:EnzymeCore.Duplicated}, tape,
        F::Union{EnzymeCore.Const, EnzymeCore.Duplicated}, b::EnzymeCore.Duplicated,
    )
    ubar = tape
    b.dval .+= F.val \ ubar   # K symmetric => adjoint solve reuses the same factorization
    fill!(ubar, 0)
    return (nothing, nothing)
end

# Same story for `splu` itself: Enzyme's activity analysis can mark its
# result Duplicated (needing a shadow for K's factorization), which forces it
# to differentiate through the Gilbert-Peierls factorization body - the same
# LLVM/MemorySSA crash as above. Since K truly never depends on A, B, that
# shadow is never actually read (our `linsolve` rule above only uses F.val),
# so we short-circuit it here too instead of letting Enzyme trace splu().
function ER.augmented_primal(
        config, func::EnzymeCore.Const{typeof(splu)}, ::Type{<:EnzymeCore.Duplicated},
        K::Union{EnzymeCore.Const, EnzymeCore.Duplicated},
    )
    F = func.val(K.val)
    return ER.AugmentedReturn(F, F, nothing)
end

function ER.reverse(
        config, func::EnzymeCore.Const{typeof(splu)}, ::Type{<:EnzymeCore.Duplicated}, tape,
        K::Union{EnzymeCore.Const, EnzymeCore.Duplicated},
    )
    return (nothing,)
end

function solve_laplace(cellvalues, dh, ch, A, B)
    K, f_global = assemble_global(cellvalues, dh, A, B)
    apply!(K, f_global, ch)
    F = splu(K)
    u = linsolve(F, f_global)
    return u
end

function solution_at(cellvalues, dh, ch, grid, A, B, x0, y0)
    u = solve_laplace(cellvalues, dh, ch, A, B)
    ph = PointEvalHandler(grid, [Vec((x0, y0))])
    return evaluate_at_points(ph, dh, u, :u)[1]
end

# Point and parameter values to differentiate at
x0, y0 = 0.3, -0.2
A0, B0 = 1.0, 1.0

grad_fd = ForwardDiff.gradient(
    p -> solution_at(cellvalues, dh, ch, grid, p[1], p[2], x0, y0),
    [A0, B0],
)
println("∂result/∂A, ∂result/∂B (ForwardDiff) = ", (grad_fd[1], grad_fd[2]))

result = Enzyme.autodiff(
    Enzyme.set_runtime_activity(Enzyme.Reverse), solution_at, Enzyme.Active,
    Enzyme.Const(cellvalues), Enzyme.Const(dh), Enzyme.Const(ch), Enzyme.Const(grid),
    Enzyme.Active(A0), Enzyme.Active(B0), Enzyme.Const(x0), Enzyme.Const(y0),
)
∂result_∂A, ∂result_∂B = result[1][4:5]
println("∂result/∂A, ∂result/∂B (Enzyme) = ", (∂result_∂A, ∂result_∂B))

u = solve_laplace(cellvalues, dh, ch, A0, B0)

VTKGridFile("heat_equation_enzyme", dh) do vtk
    write_solution(vtk, dh, u)
end
