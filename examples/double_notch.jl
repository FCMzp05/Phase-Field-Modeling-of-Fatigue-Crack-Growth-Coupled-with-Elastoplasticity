#!/usr/bin/env julia
# ============================================================================
# Asymmetric double-notch tension (ADNT) specimen
# Reproduces Section 4.3 of the manuscript.
# ============================================================================

using Gmsh: gmsh
using Gridap, Gridap.Geometry, Gridap.TensorValues, Gridap.CellData
using Gridap.FESpaces, Gridap.Algebra, Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using GridapGmsh, LinearAlgebra, Statistics, Printf
ENV["GKSwstype"] = "100"
using Plots; gr(show=false)
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "ElastoPlasticFatiguePF.jl"))
using .ElastoPlasticFatiguePF

# ============================================================================
# Parameters
# ============================================================================
const W_sp = 18.0; const H_sp = 50.0; const r_notch = 2.5
const LEFT_NOTCH_X = 0.0; const LEFT_NOTCH_Y = 30.0
const RIGHT_NOTCH_X = W_sp; const RIGHT_NOTCH_Y = 20.0

const mat = MaterialData(213000.0, 0.33, 100.0, 0.1)
const C_tensor = elastic_stiffness_2D(mat)

const l0 = 0.3; const gc = 54.0; const αT = 100.0
const η_stab = 1e-10; const η_plastic = 1.0

const u_max = 0.05; const u_min = 0.0; const R_load = 0.0

const N_CYCLES_TOTAL = 300; const N_STEPS_PER_CYCLE = 200; const CYCLE_JUMP = 1
const phase_every = 1; const max_stagger = 20; const stagger_tol = 1e-6

const h_coarse = 3.0; const h_fine = 0.3
const OUTPUT_DIR = "results_ADNT"; const FORCE_GC_EVERY = 20

# ============================================================================
# Mesh
# ============================================================================
function create_mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ADNT")

    N_ARC = 40
    left_pts = Int[]
    for i in 0:N_ARC
        θ = π/2 - i * π / N_ARC
        x = LEFT_NOTCH_X + r_notch * cos(θ)
        y = LEFT_NOTCH_Y + r_notch * sin(θ)
        h = (x < r_notch) ? h_fine : h_coarse
        push!(left_pts, gmsh.model.geo.addPoint(x, y, 0, h))
    end
    right_pts = Int[]
    for i in 0:N_ARC
        θ = π/2 - i * π / N_ARC
        x = RIGHT_NOTCH_X - r_notch * cos(θ)
        y = RIGHT_NOTCH_Y + r_notch * sin(θ)
        h = (W_sp - x < r_notch) ? h_fine : h_coarse
        push!(right_pts, gmsh.model.geo.addPoint(x, y, 0, h))
    end

    p_bl = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
    p_br = gmsh.model.geo.addPoint(W_sp, 0, 0, h_coarse)
    p_tr = gmsh.model.geo.addPoint(W_sp, H_sp, 0, h_coarse)
    p_tl = gmsh.model.geo.addPoint(0, H_sp, 0, h_coarse)

    lines = Int[]
    push!(lines, gmsh.model.geo.addLine(p_bl, p_br))
    push!(lines, gmsh.model.geo.addLine(p_br, right_pts[1]))
    for i in 1:N_ARC
        push!(lines, gmsh.model.geo.addLine(right_pts[i], right_pts[i+1]))
    end
    push!(lines, gmsh.model.geo.addLine(right_pts[end], p_tr))
    push!(lines, gmsh.model.geo.addLine(p_tr, p_tl))
    push!(lines, gmsh.model.geo.addLine(p_tl, left_pts[1]))
    for i in 1:N_ARC
        push!(lines, gmsh.model.geo.addLine(left_pts[i], left_pts[i+1]))
    end
    push!(lines, gmsh.model.geo.addLine(left_pts[end], p_bl))

    cl = gmsh.model.geo.addCurveLoop(lines)
    sf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [sf], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, [lines[1]], 1)
    gmsh.model.setPhysicalName(1, 1, "bottom")
    push!(lines, -1)  # sentinel
    top_idx = 2 + N_ARC + 2
    gmsh.model.addPhysicalGroup(1, [lines[top_idx]], 2)
    gmsh.model.setPhysicalName(1, 2, "top")

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0)
    gmsh.model.mesh.field.setNumber(1, "XMax", W_sp)
    gmsh.model.mesh.field.setNumber(1, "YMin", RIGHT_NOTCH_Y - r_notch - 2)
    gmsh.model.mesh.field.setNumber(1, "YMax", LEFT_NOTCH_Y + r_notch + 2)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)

    mf = joinpath(OUTPUT_DIR, "mesh.msh"); gmsh.write(mf); gmsh.finalize()
    return mf
end

# ============================================================================
# Simulation driver
# ============================================================================
function run_simulation()
    mkpath(OUTPUT_DIR)
    println("\n" * "="^70)
    println("  ADNT specimen")
    println("="^70)

    mesh_file = create_mesh()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["bottom","top"],
                      dirichlet_masks=[(true,true),(false,true)])
    u_bottom(x) = VectorValue(0.0, 0.0)
    u_top(x, u) = VectorValue(0.0, u)
    U_u = TrialFESpace(V_u)

    V_s = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
    U_s = TrialFESpace(V_s)
    V_dg0 = FESpace(model, ReferenceFE(lagrangian, Float64, 0), conformity=:L2)
    V_L2  = FESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:L2)

    qc = CellQuadrature(Ω, 2); cpc = get_cell_points(qc)
    nc = num_cells(Ω)
    println("  Elements: $(nc)")

    plastic_states = [PlasticStateData() for _ in 1:nc]
    H_prev = CellState(0.0, dΩ)
    ᾱ_acc = zeros(nc); α_prev = zeros(nc)
    sh = interpolate_everywhere(x->0.0, U_s); uh = zero(U_u)

    n_half = N_STEPS_PER_CYCLE ÷ 2
    load_steps   = range(u_min, u_max, length=n_half+1)
    unload_steps = range(u_max, u_min, length=n_half+1)[2:end]
    cycle_hist = Int[]; u_prev_val = u_min; ψ_t = zeros(nc); cur = 0

    cell_coords = get_cell_coordinates(Ω)

    while cur < N_CYCLES_TOTAL
        cur += 1
        for (_, ur) in [("load", load_steps), ("unload", unload_steps)]
            for uv in ur
                U_u = TrialFESpace(V_u, [u_bottom, x->u_top(x,uv)])
                try
                    uh = solve_displacement(sh, plastic_states, uv,
                            V_u, U_u, dΩ, Ω, uh, u_prev_val,
                            cpc, mat, C_tensor, η_stab, u_max)
                catch; break; end
                ε_qp = evaluate(ε(uh), cpc); s_qp = evaluate(sh, cpc)
                for i in 1:nc
                    ψp = plastic_strain_energy(mat, plastic_states[i].εp_eq)
                    mx = 0.0
                    for q in 1:length(ε_qp[i])
                        mx = max(mx, tensile_elastic_strain_energy(
                                     ε_qp[i][q], plastic_states[i], mat))
                    end
                    ψ_t[i] = mx + η_plastic*ψp
                end
                update_state!(update_history_variable, H_prev,
                              FEFunction(V_dg0, ψ_t))
                for i in 1:nc
                    gs = (1.0-clamp(s_qp[i][1],0.0,1.0))^2 + η_stab
                    αi = gs*ψ_t[i]; ᾱ_acc[i] += max(αi-α_prev[i],0.0)
                    α_prev[i] = αi
                end
                if phase_every > 0
                    for _ in 1:max_stagger
                        so = copy(get_free_dof_values(sh))
                        Hf = project_to_fe_space(H_prev, V_L2, dΩ)
                        af = array_to_fe_field(ᾱ_acc, V_dg0, V_L2, dΩ)
                        sh = solve_phase_field(sh, Hf, af, V_s, U_s, dΩ;
                                               Gc=gc, l0=l0, αT=αT)
                        norm(get_free_dof_values(sh)-so)/(norm(get_free_dof_values(sh))+1e-10) < stagger_tol && break
                    end
                end
                u_prev_val = uv
            end
        end
        push!(cycle_hist, cur)

        if cur % 5 == 0
            max_s = maximum(get_free_dof_values(sh))
            @printf("  N=%4d  max(s)=%.4f\n", cur, max_s)
            cycle_str = lpad(cur, 5, '0')
            writevtk(Ω, joinpath(OUTPUT_DIR, "result_N$(cycle_str)"),
                     cellfields=["d"=>sh, "u"=>uh], nsubcells=4)
        end

        cur % FORCE_GC_EVERY == 0 && GC.gc(false)
    end

    println("  Done: $(cur) cycles")
    return cycle_hist
end

function main()
    run_simulation()
end

main()
