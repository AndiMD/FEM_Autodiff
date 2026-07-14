using Ferrite, SparseArrays, ForwardDiff, PureUMFPACK

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

function assemble_element!(Ke::AbstractMatrix, fe::AbstractVector, cellvalues::CellValues, coords::Vector{<:Vec}, A, B)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Physical coordinate of this quadrature point
        x = spatial_coordinate(cellvalues, q_point, coords)
        # Source term, depends on the parameters A and B
        f_qp = A * sin(x[1]) + B * cos(x[2])
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += f_qp * δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues::CellValues, dh::DofHandler, A, B)
    # K's numerical values never actually depend on A, B, so it is assembled
    # as plain Float64 and factorized directly as a sparse matrix. Only fe /
    # f_global carry the promoted type of A, B (e.g. ForwardDiff.Dual when
    # differentiating).
    n_basefuncs = getnbasefunctions(cellvalues)
    T = promote_type(typeof(A), typeof(B))
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(T, n_basefuncs)
    K = allocate_matrix(dh)
    f_global = zeros(T, ndofs(dh))
    # Create an assembler (matrix-only; f_global is assembled separately
    # since its element type may differ from K's)
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element!(Ke, fe, cellvalues, getcoordinates(cell), A, B)
        # Assemble Ke into K and fe into f_global
        assemble!(assembler, celldofs(cell), Ke)
        assemble!(f_global, celldofs(cell), fe)
    end
    return K, f_global
end

function solve_laplace(A, B)
    K, f_global = assemble_global(cellvalues, dh, A, B)
    apply!(K, f_global, ch)
    # PureUMFPACK's sparse LU solve is implemented as plain generic-arithmetic
    # loops (no C/BLAS calls requiring Float64), so `F \ f_global` stays
    # differentiable even when f_global holds ForwardDiff.Dual entries -
    # unlike SparseArrays' UMFPACK wrapper, which requires Float64 throughout.
    F = splu(K)
    u = F \ f_global
    return u
end

function solution_at(A, B, x0, y0)
    u = solve_laplace(A, B)
    ph = PointEvalHandler(grid, [Vec((x0, y0))])
    return evaluate_at_points(ph, dh, u, :u)[1]
end

# Point and parameter values to differentiate at
x0, y0 = 0.3, -0.2
A0, B0 = 1.0, 1.0

∂result_∂A, ∂result_∂B = ForwardDiff.gradient(p -> solution_at(p[1], p[2], x0, y0), [A0, B0])

u = solve_laplace(A0, B0)

VTKGridFile("heat_equation", dh) do vtk
    write_solution(vtk, dh, u)
end
