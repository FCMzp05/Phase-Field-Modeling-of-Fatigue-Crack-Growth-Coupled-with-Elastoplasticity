# Displacement solver with Picard iteration for elastoplastic equilibrium.

"""
    compute_stress!(uh, sh, plastic_states, mat, η_stab, Ω, cell_points,
                    update_state) -> Vector{SymTensorValue{2}}

Evaluate the constitutive response at each cell centroid.
"""
function compute_stress!(
    uh, sh,
    plastic_states::Vector{PlasticStateData},
    mat::MaterialData,
    η_stab::Float64,
    Ω, cell_points,
    update_state::Bool
)
    εh = ε(uh)
    ε_at_qp = evaluate(εh, cell_points)
    s_at_qp = evaluate(sh, cell_points)

    n_cells  = num_cells(Ω)
    σ_values = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)

    for i in 1:n_cells
        ε_cell = ε_at_qp[i][1]
        s_cell = clamp(s_at_qp[i][1], 0.0, 1.0)
        σ_values[i] = return_mapping!(ε_cell, plastic_states[i],
                                      mat, s_cell, η_stab,
                                      update_state=update_state)
    end

    return σ_values
end

"""
    solve_displacement(sh, plastic_states, u_val, V_u, U_u, dΩ, Ω,
                       uh_prev, u_prev, cell_points, mat, C_tensor,
                       η_stab, u_max) -> FEFunction

Picard iteration for the nonlinear displacement problem.
"""
function solve_displacement(
    sh, plastic_states,
    u_val::Float64, V_u, U_u, dΩ, Ω,
    uh_prev, u_prev::Float64,
    cell_points,
    mat::MaterialData,
    C_tensor,
    η_stab::Float64,
    u_max::Float64
)
    g_s(s) = (1.0 - s)^2 + η_stab
    n_cells = num_cells(Ω)
    plastic_work = [copy_plastic_state(ps) for ps in plastic_states]

    Δu = u_val - u_prev
    n_substeps = max(1, ceil(Int, abs(Δu) / (u_max / 10.0)))
    uh = uh_prev

    for _ in 1:n_substeps
        plastic_start = [copy_plastic_state(ps) for ps in plastic_work]

        for _ in 1:10
            u_old = copy(get_free_dof_values(uh))

            σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, n_cells)
            for i in 1:n_cells
                σp_in = C_tensor ⊙ plastic_work[i].εp
                add   = mat.λ * plastic_work[i].εp33
                σp_vals[i] = σp_in + SymTensorValue{2,Float64}(add, 0.0, add)
            end
            σp_field = CellField(σp_vals, Ω)

            a(u, v) = ∫((g_s ∘ sh) * (C_tensor ⊙ ε(u)) ⊙ ε(v)) * dΩ
            l(v)    = ∫((g_s ∘ sh) * σp_field ⊙ ε(v)) * dΩ

            op = AffineFEOperator(a, l, U_u, V_u)
            uh = solve(op)

            for i in 1:n_cells
                plastic_work[i].εp    = plastic_start[i].εp
                plastic_work[i].εp33  = plastic_start[i].εp33
                plastic_work[i].εp_eq = plastic_start[i].εp_eq
            end
            compute_stress!(uh, sh, plastic_work, mat, η_stab,
                            Ω, cell_points, true)

            u_change = norm(get_free_dof_values(uh) - u_old) /
                       (norm(get_free_dof_values(uh)) + 1e-10)
            u_change < 1e-4 && break
        end
    end

    for i in 1:n_cells
        plastic_states[i].εp    = plastic_work[i].εp
        plastic_states[i].εp33  = plastic_work[i].εp33
        plastic_states[i].εp_eq = plastic_work[i].εp_eq
    end

    return uh
end
