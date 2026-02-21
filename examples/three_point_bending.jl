#!/usr/bin/env julia
# ============================================================================
# Three-point bending (3PB) fatigue specimen
# Reproduces Fig. 22 (Section 4.2) of the manuscript.
# ============================================================================

using Gmsh: gmsh
using Gridap, Gridap.Geometry, Gridap.TensorValues, Gridap.CellData
using Gridap.FESpaces, Gridap.Algebra, Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using GridapGmsh, LinearAlgebra, Statistics, Printf, Plots; gr()
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "ElastoPlasticFatiguePF.jl"))
using .ElastoPlasticFatiguePF

# ============================================================================
# Parameters
# ============================================================================
const W = 50.0;  const H_beam = 10.0
const x_support_L = 10.0; const x_support_R = 40.0; const x_load = 25.0
const support_halfwidth = 0.5; const load_halfwidth = 0.5
const notch_depth = 1.0; const notch_halfwidth = 0.05

const mat = MaterialData(210000.0, 0.3, 440.0, 0.1)
const C_tensor = elastic_stiffness_2D(mat)

const l0 = 0.1; const Gc = 2.7; const αT = 2.25
const η_stab = 1e-6; const η_plastic = 1.0

const u_max = 0.02; const R_load = 0.0; const u_min = 0.0

const N_CYCLES = 4000; const N_STEPS_PER_CYCLE = 20
const phase_every = 1; const max_stagger = 20; const stagger_tol = 1e-6
const h_fine = 0.08; const h_coarse = 1.0
const OUTPUT_DIR = "results_3PB"; const FORCE_GC_EVERY = 20

# ============================================================================
# Mesh
# ============================================================================
function create_mesh()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("3PB")

    p1 = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
    p2 = gmsh.model.geo.addPoint(x_load - notch_halfwidth, 0, 0, h_fine)
    p3 = gmsh.model.geo.addPoint(x_load - notch_halfwidth, notch_depth, 0, h_fine)
    p4 = gmsh.model.geo.addPoint(x_load + notch_halfwidth, notch_depth, 0, h_fine)
    p5 = gmsh.model.geo.addPoint(x_load + notch_halfwidth, 0, 0, h_fine)
    p6 = gmsh.model.geo.addPoint(W, 0, 0, h_coarse)
    p7 = gmsh.model.geo.addPoint(W, H_beam, 0, h_coarse)
    p8 = gmsh.model.geo.addPoint(x_load + load_halfwidth, H_beam, 0, h_fine)
    p9 = gmsh.model.geo.addPoint(x_load - load_halfwidth, H_beam, 0, h_fine)
    p10 = gmsh.model.geo.addPoint(0, H_beam, 0, h_coarse)

    l1  = gmsh.model.geo.addLine(p1, p2)
    l2  = gmsh.model.geo.addLine(p2, p3)
    l3  = gmsh.model.geo.addLine(p3, p4)
    l4  = gmsh.model.geo.addLine(p4, p5)
    l5  = gmsh.model.geo.addLine(p5, p6)
    l6  = gmsh.model.geo.addLine(p6, p7)
    l7  = gmsh.model.geo.addLine(p7, p8)
    l8  = gmsh.model.geo.addLine(p8, p9)
    l9  = gmsh.model.geo.addLine(p9, p10)
    l10 = gmsh.model.geo.addLine(p10, p1)

    cl = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6,l7,l8,l9,l10])
    sf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2,[sf],1)
    gmsh.model.setPhysicalName(2,1,"Domain")
    gmsh.model.addPhysicalGroup(1,[l1,l5],1)
    gmsh.model.setPhysicalName(1,1,"bottom")
    gmsh.model.addPhysicalGroup(1,[l8],2)
    gmsh.model.setPhysicalName(1,2,"load_top")
    gmsh.model.addPhysicalGroup(1,[l7,l9],3)
    gmsh.model.setPhysicalName(1,3,"top_side")

    gmsh.model.mesh.field.add("Box",1)
    gmsh.model.mesh.field.setNumber(1,"VIn",h_fine)
    gmsh.model.mesh.field.setNumber(1,"VOut",h_coarse)
    gmsh.model.mesh.field.setNumber(1,"XMin",x_load-3)
    gmsh.model.mesh.field.setNumber(1,"XMax",x_load+3)
    gmsh.model.mesh.field.setNumber(1,"YMin",0)
    gmsh.model.mesh.field.setNumber(1,"YMax",H_beam)
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
    println("  Three-point bending -- Fig. 22")
    println("="^70)

    mesh_file = create_mesh()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)

    function is_left_support(x)
        abs(x[1]-x_support_L) < support_halfwidth && abs(x[2]) < 1e-8
    end
    function is_right_support(x)
        abs(x[1]-x_support_R) < support_halfwidth && abs(x[2]) < 1e-8
    end

    V_u = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["bottom","load_top"],
                      dirichlet_masks=[(false,false),(false,true)])

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
    sh = interpolate_everywhere(x->0.0, U_s)

    u_bot(x)    = VectorValue(0.0, 0.0)
    u_load(x,u) = VectorValue(0.0, -u)

    U_u = TrialFESpace(V_u, [u_bot, x->u_load(x,0.0)])
    uh = zero(U_u)

    n_half = N_STEPS_PER_CYCLE ÷ 2
    load_steps   = range(0.0, u_max, length=n_half+1)
    unload_steps = range(u_max, 0.0, length=n_half+1)[2:end]
    cycle_hist = Int[]; crack_hist = Float64[]
    u_prev_val = 0.0; ψ_t = zeros(nc); cur = 0

    cell_coords = get_cell_coordinates(Ω)
    println("  Starting fatigue cycling...")

    while cur < N_CYCLES
        cur += 1
        for (_, ur) in [("load", load_steps), ("unload", unload_steps)]
            for uv in ur
                U_u = TrialFESpace(V_u, [u_bot, x->u_load(x,uv)])
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
                for _ in 1:max_stagger
                    so = copy(get_free_dof_values(sh))
                    Hf = project_to_fe_space(H_prev, V_L2, dΩ)
                    af = array_to_fe_field(ᾱ_acc, V_dg0, V_L2, dΩ)
                    sh = solve_phase_field(sh, Hf, af, V_s, U_s, dΩ;
                                           Gc=Gc, l0=l0, αT=αT)
                    norm(get_free_dof_values(sh)-so)/(norm(get_free_dof_values(sh))+1e-10) < stagger_tol && break
                end
                u_prev_val = uv
            end
        end

        # Track crack tip (vertical direction from notch tip)
        coords_n = get_node_coordinates(Ω)
        s_vals   = get_free_dof_values(sh)
        y_max    = notch_depth
        for i in eachindex(s_vals)
            x, y = coords_n[i][1], coords_n[i][2]
            if abs(x - x_load) < 2*l0 && y > notch_depth - l0 && s_vals[i] > 0.5
                y_max = max(y_max, y)
            end
        end
        Δa = y_max - notch_depth
        push!(cycle_hist, cur); push!(crack_hist, Δa)

        if cur % 50 == 0
            max_s = maximum(get_free_dof_values(sh))
            @printf("  N=%5d  max(s)=%.4f  Δa=%.4f mm\n", cur, max_s, Δa)
        end

        cur % FORCE_GC_EVERY == 0 && GC.gc(false)
    end

    writedlm(joinpath(OUTPUT_DIR, "data.csv"), hcat(cycle_hist, crack_hist), ',')
    println("  Done: $(cur) cycles")
    return cycle_hist, crack_hist
end

function main()
    cycles, lengths = run_simulation()
    if !isempty(cycles)
        p = plot(cycles, lengths, xlabel="N", ylabel="Δa (mm)",
                 title="3PB Fatigue Crack Growth", label="Simulation",
                 linewidth=2, grid=true, size=(800,600), dpi=300)
        savefig(p, joinpath(OUTPUT_DIR, "crack_growth.png"))
    end
end

main()
