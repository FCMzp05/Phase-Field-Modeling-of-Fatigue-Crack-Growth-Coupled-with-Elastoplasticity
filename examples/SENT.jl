#!/usr/bin/env julia
# ============================================================================
# SENT specimen -- baseline fatigue crack growth
# Reproduces Fig. 9a (Section 4.1.2) of the manuscript.
# ============================================================================

using Gmsh: gmsh
using Gridap
using Gridap.Geometry
using Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using Gridap.TensorValues
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Algebra
using GridapGmsh
using LinearAlgebra
using Statistics
using Printf
using Plots; gr()
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "ElastoPlasticFatiguePF.jl"))
using .ElastoPlasticFatiguePF

# ============================================================================
# Parameters
# ============================================================================

# Geometry [mm]
const W  = 1.0        # specimen width
const H  = 1.0        # specimen height
const a0 = 0.5        # initial crack length

# Material (Table 3)
const mat = MaterialData(210000.0, 0.3, 440.0, 0.1)
const C_tensor = elastic_stiffness_2D(mat)

# Phase-field / fatigue (Table 3)
const l0         = 0.02     # regularisation length [mm]
const Gc         = 2.7      # fracture toughness [N/mm]
const αT         = 11.25    # fatigue threshold [N/mm^2]
const η_stab     = 1e-6     # residual stiffness
const η_plastic  = 1.0      # Taylor--Quinney weight

# Loading (Section 4.1.2)
const u_max  = 0.001   # displacement amplitude [mm]
const R_load = 0.0     # load ratio
const u_min  = R_load * u_max

# Cycling
const N_CYCLES          = 800
const N_STEPS_PER_CYCLE = 20
const phase_every       = 1
const max_stagger       = 20
const stagger_tol       = 1e-6

# Mesh
const h_fine   = 0.005    # crack-path refinement [mm]
const h_coarse = 0.05     # far-field element size [mm]

# Output
const OUTPUT_DIR    = "results_SENT"
const ENABLE_PNG    = true
const SAVE_PNG_EVERY = 10
const PNG_DPI       = 110
const FORCE_GC_EVERY = 20

# ============================================================================
# Mesh generation
# ============================================================================

function create_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("SENT")

    y_crack = H / 2.0
    CH = h_fine

    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_coarse, 1)
    gmsh.model.geo.addPoint(W, 0.0, 0.0, h_coarse, 2)
    gmsh.model.geo.addPoint(W, H/2, 0.0, h_coarse, 22)
    gmsh.model.geo.addPoint(W, H, 0.0, h_coarse, 3)
    gmsh.model.geo.addPoint(0.0, H, 0.0, h_coarse, 4)
    gmsh.model.geo.addPoint(0.0, y_crack + 0.5*CH, 0.0, h_coarse, 5)
    gmsh.model.geo.addPoint(a0 - CH, y_crack + 0.5*CH, 0.0, h_fine, 6)
    gmsh.model.geo.addPoint(a0, y_crack, 0.0, h_fine, 7)
    gmsh.model.geo.addPoint(a0 - CH, y_crack - 0.5*CH, 0.0, h_fine, 8)
    gmsh.model.geo.addPoint(0.0, y_crack - 0.5*CH, 0.0, h_coarse, 9)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 22, 11)
    gmsh.model.geo.addLine(22, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 7, 6)
    gmsh.model.geo.addLine(7, 8, 7)
    gmsh.model.geo.addLine(8, 9, 8)
    gmsh.model.geo.addLine(9, 1, 9)

    gmsh.model.geo.addCurveLoop([1, 11, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, [1], 1)
    gmsh.model.setPhysicalName(1, 1, "bottom")
    gmsh.model.addPhysicalGroup(1, [3], 2)
    gmsh.model.setPhysicalName(1, 2, "top")

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", a0 - 0.1)
    gmsh.model.mesh.field.setNumber(1, "XMax", W)
    gmsh.model.mesh.field.setNumber(1, "YMin", y_crack - 0.2)
    gmsh.model.mesh.field.setNumber(1, "YMax", y_crack + 0.2)

    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)

    mesh_file = joinpath(OUTPUT_DIR, "mesh.msh")
    gmsh.write(mesh_file)
    gmsh.finalize()
    return mesh_file
end

# ============================================================================
# Simulation driver
# ============================================================================

function run_simulation()
    mkpath(OUTPUT_DIR)
    println("\n" * "="^70)
    println("  SENT baseline -- Fig. 9a")
    println("="^70)

    mesh_file = create_mesh()
    model = GmshDiscreteModel(mesh_file)
    writevtk(model, joinpath(OUTPUT_DIR, "mesh"))
    plot_mesh_png(model, joinpath(OUTPUT_DIR, "mesh.png"); title="SENT Mesh")

    Ω  = Triangulation(model)
    dΩ = Measure(Ω, 2)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["bottom", "top"],
                      dirichlet_masks=[(true, true), (false, true)])
    u_bottom(x) = VectorValue(0.0, 0.0)
    u_top(x, u) = VectorValue(0.0, u)
    U_u = TrialFESpace(V_u)

    reffe_s = ReferenceFE(lagrangian, Float64, 1)
    V_s = TestFESpace(model, reffe_s, conformity=:H1)
    U_s = TrialFESpace(V_s)

    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = FESpace(model, reffe_dg0, conformity=:L2)
    V_L2  = FESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

    quad_cache       = CellQuadrature(Ω, 2)
    cell_points_cache = get_cell_points(quad_cache)

    n_cells = num_cells(Ω)
    println("  Elements: $(n_cells)")

    # State initialisation
    plastic_states = [PlasticStateData() for _ in 1:n_cells]
    H_prev         = CellState(0.0, dΩ)
    ᾱ_accumulated  = zeros(Float64, n_cells)
    α_inst_prev    = zeros(Float64, n_cells)

    sh = interpolate_everywhere(x -> 0.0, U_s)
    uh = zero(U_u)

    cycle_history = Int[]
    crack_history = Float64[]

    n_half       = N_STEPS_PER_CYCLE ÷ 2
    load_steps   = range(u_min, u_max, length=n_half + 1)
    unload_steps = range(u_max, u_min, length=n_half + 1)[2:end]

    current_cycle = 0
    u_prev_val    = u_min
    cell_coords   = get_cell_coordinates(Ω)
    ψ_elastic     = zeros(n_cells)
    ψ_total       = zeros(n_cells)

    println("  Starting fatigue cycling...")

    while current_cycle < N_CYCLES
        current_cycle += 1

        for (_, u_range) in [("load", load_steps), ("unload", unload_steps)]
            for u_val in u_range
                U_u = TrialFESpace(V_u, [u_bottom, x -> u_top(x, u_val)])

                try
                    uh = solve_displacement(sh, plastic_states, u_val,
                            V_u, U_u, dΩ, Ω, uh, u_prev_val,
                            cell_points_cache, mat, C_tensor, η_stab, u_max)
                catch e
                    isa(e, SingularException) && (@goto sim_end)
                    rethrow(e)
                end

                εh = ε(uh)
                ε_qp = evaluate(εh, cell_points_cache)
                s_qp = evaluate(sh, cell_points_cache)

                for i in 1:n_cells
                    ψ_p   = plastic_strain_energy(mat, plastic_states[i].εp_eq)
                    ψ_max = 0.0
                    for q in 1:length(ε_qp[i])
                        ψe_q = tensile_elastic_strain_energy(
                                    ε_qp[i][q], plastic_states[i], mat)
                        ψ_max = max(ψ_max, ψe_q)
                    end
                    ψ_elastic[i] = ψ_max
                    ψ_total[i]   = ψ_max + η_plastic * ψ_p
                end

                ψ_total_dg = FEFunction(V_dg0, ψ_total)
                update_state!(update_history_variable, H_prev, ψ_total_dg)

                for i in 1:n_cells
                    s_cell = clamp(s_qp[i][1], 0.0, 1.0)
                    g_s    = (1.0 - s_cell)^2 + η_stab
                    α_inst = g_s * ψ_total[i]
                    Δᾱ     = max(α_inst - α_inst_prev[i], 0.0)
                    ᾱ_accumulated[i] += Δᾱ
                    α_inst_prev[i]    = α_inst
                end

                if phase_every > 0
                    for _ in 1:max_stagger
                        s_old   = copy(get_free_dof_values(sh))
                        H_field = project_to_fe_space(H_prev, V_L2, dΩ)
                        ᾱ_field = array_to_fe_field(ᾱ_accumulated, V_dg0, V_L2, dΩ)
                        sh = solve_phase_field(sh, H_field, ᾱ_field, V_s, U_s, dΩ;
                                               Gc=Gc, l0=l0, αT=αT)
                        res = norm(get_free_dof_values(sh) - s_old) /
                              (norm(get_free_dof_values(sh)) + 1e-10)
                        res < stagger_tol && break
                    end
                end

                u_prev_val = u_val
            end
        end

        a_current = track_crack_tip_x(sh, Ω, a0, H/2.0, l0)
        Δa = a_current - a0
        push!(cycle_history, current_cycle)
        push!(crack_history, Δa)

        if current_cycle % 10 == 0
            max_s = maximum(get_free_dof_values(sh))
            @printf("  N=%4d  max(s)=%.4f  Δa=%.4f mm\n",
                    current_cycle, max_s, Δa)
        end

        if ENABLE_PNG && current_cycle % SAVE_PNG_EVERY == 0
            s_cell_vals = zeros(n_cells)
            s_at = evaluate(sh, cell_points_cache)
            for i in 1:n_cells
                nqp = length(s_at[i]); acc = 0.0
                for q in 1:nqp; acc += clamp(s_at[i][q], 0.0, 1.0); end
                s_cell_vals[i] = acc / max(nqp, 1)
            end
            cycle_str = lpad(current_cycle, 6, '0')
            plot_cell_scalar_png(cell_coords, s_cell_vals;
                title="Damage | N=$(current_cycle), Δa=$(round(Δa, digits=3)) mm",
                out_file=joinpath(OUTPUT_DIR, "damage_N$(cycle_str).png"),
                dpi=PNG_DPI, cmap=:jet, vmin=0.0, vmax=1.0)
        end

        if current_cycle % 10 == 0
            cycle_str = lpad(current_cycle, 6, '0')
            εp_eq_vals = [ps.εp_eq for ps in plastic_states]
            εp_eq_cf   = CellField(εp_eq_vals, Ω)
            αbar_cf    = CellField(ᾱ_accumulated, Ω)
            writevtk(Ω, joinpath(OUTPUT_DIR, "result_N$(cycle_str)"),
                     cellfields=["d"=>sh, "u"=>uh,
                                 "epspeq"=>εp_eq_cf,
                                 "alpha_bar"=>αbar_cf],
                     nsubcells=4)
        end

        current_cycle % FORCE_GC_EVERY == 0 && GC.gc(false)

        (Δa > 0.4 || a_current > 0.95) && break
    end

    @label sim_end
    println("  Done: $(current_cycle) cycles, Δa=$(round(crack_history[end], digits=4)) mm")

    writedlm(joinpath(OUTPUT_DIR, "data.csv"), hcat(cycle_history, crack_history), ',')
    return cycle_history, crack_history
end

# ============================================================================
# Entry point
# ============================================================================

function main()
    cycles, lengths = run_simulation()
    if !isempty(cycles)
        p = plot(cycles, lengths,
                 xlabel="Number of cycles N",
                 ylabel="Crack extension Δa (mm)",
                 title="SENT Fatigue Crack Growth",
                 label="Simulation", linewidth=2, grid=true,
                 size=(800,600), dpi=300)
        savefig(p, joinpath(OUTPUT_DIR, "crack_growth.png"))
    end
end

main()
