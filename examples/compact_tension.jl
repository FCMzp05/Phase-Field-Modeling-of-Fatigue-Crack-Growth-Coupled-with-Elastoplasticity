#!/usr/bin/env julia
# ============================================================================
# Compact tension (CT) specimen
# Reproduces Fig. 9 from Xie et al. (2023) using the Jia-style coupling.
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
# CT specimen geometry [mm]
const W_CT = 50.0; const H_CT = 48.0; const THICKNESS = 25.0
const PIN_RADIUS = 5.0
const PIN1_X = 8.0; const PIN1_Y = 13.0
const PIN2_X = 8.0; const PIN2_Y = 35.0
const Y_CENTER = 24.0
const V_TIP_X = 14.0; const V_TIP_Y = Y_CENTER
const V_ANGLE = 30.0; const V_BASE_WIDTH = 2.0
const V_HALF_WIDTH = V_BASE_WIDTH / 2.0
const V_HALF_ANGLE_RAD = deg2rad(V_ANGLE / 2.0)
const V_DEPTH = V_HALF_WIDTH / tan(V_HALF_ANGLE_RAD)
const RECT_LENGTH = V_TIP_X - V_DEPTH

# Material (GH4169 @ 550 deg C)
const mat = MaterialData(1.61e5, 0.3, 900.0, 0.1)
const C_tensor = elastic_stiffness_2D(mat)

const l0 = 0.4; const GcI = 47.5; const GcII = 47.5
const η_stab = 1e-4; const αT = 5000.0

const F_MAX_3D = 25000.0
const F_MAX = F_MAX_3D / THICKNESS
const R_LOAD = 0.1; const F_MIN = R_LOAD * F_MAX

const N_CYCLES_TOTAL = 60000; const N_STEPS_PER_CYCLE = 20
const CYCLE_JUMP = 100; const phase_every = 1
const max_stagger = 20; const stagger_tol = 1e-6

const h_fine = 0.2; const h_coarse = 2.0
const OUTPUT_DIR = "results_CT"; const FORCE_GC_EVERY = 20

# ============================================================================
# Mesh
# ============================================================================
function create_CT_mesh()
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CT")

    # Outer rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
    p2 = gmsh.model.geo.addPoint(W_CT, 0, 0, h_coarse)
    p3 = gmsh.model.geo.addPoint(W_CT, H_CT, 0, h_coarse)
    p4 = gmsh.model.geo.addPoint(0, H_CT, 0, h_coarse)

    # V-notch
    pv1 = gmsh.model.geo.addPoint(0, Y_CENTER - V_HALF_WIDTH, 0, h_fine)
    pv2 = gmsh.model.geo.addPoint(RECT_LENGTH, Y_CENTER - V_HALF_WIDTH, 0, h_fine)
    pv3 = gmsh.model.geo.addPoint(V_TIP_X, V_TIP_Y, 0, h_fine)
    pv4 = gmsh.model.geo.addPoint(RECT_LENGTH, Y_CENTER + V_HALF_WIDTH, 0, h_fine)
    pv5 = gmsh.model.geo.addPoint(0, Y_CENTER + V_HALF_WIDTH, 0, h_fine)

    # Outer boundary
    l1  = gmsh.model.geo.addLine(p1, p2)
    l2  = gmsh.model.geo.addLine(p2, p3)
    l3  = gmsh.model.geo.addLine(p3, p4)
    l4  = gmsh.model.geo.addLine(p4, pv5)
    l5  = gmsh.model.geo.addLine(pv5, pv4)
    l6  = gmsh.model.geo.addLine(pv4, pv3)
    l7  = gmsh.model.geo.addLine(pv3, pv2)
    l8  = gmsh.model.geo.addLine(pv2, pv1)
    l9  = gmsh.model.geo.addLine(pv1, p1)

    # Pin holes
    pc1 = gmsh.model.geo.addPoint(PIN1_X, PIN1_Y, 0, h_fine)
    arc1_pts = Int[]
    for i in 0:20
        θ = 2π * i / 20
        push!(arc1_pts, gmsh.model.geo.addPoint(
            PIN1_X + PIN_RADIUS*cos(θ), PIN1_Y + PIN_RADIUS*sin(θ), 0, h_fine))
    end
    pin1_lines = Int[]
    for i in 1:20
        push!(pin1_lines, gmsh.model.geo.addLine(arc1_pts[i], arc1_pts[i+1]))
    end
    pin1_cl = gmsh.model.geo.addCurveLoop(pin1_lines)

    pc2 = gmsh.model.geo.addPoint(PIN2_X, PIN2_Y, 0, h_fine)
    arc2_pts = Int[]
    for i in 0:20
        θ = 2π * i / 20
        push!(arc2_pts, gmsh.model.geo.addPoint(
            PIN2_X + PIN_RADIUS*cos(θ), PIN2_Y + PIN_RADIUS*sin(θ), 0, h_fine))
    end
    pin2_lines = Int[]
    for i in 1:20
        push!(pin2_lines, gmsh.model.geo.addLine(arc2_pts[i], arc2_pts[i+1]))
    end
    pin2_cl = gmsh.model.geo.addCurveLoop(pin2_lines)

    outer_cl = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4,l5,l6,l7,l8,l9])
    sf = gmsh.model.geo.addPlaneSurface([outer_cl, pin1_cl, pin2_cl])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [sf], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    gmsh.model.addPhysicalGroup(1, pin1_lines, 1)
    gmsh.model.setPhysicalName(1, 1, "pin_lower")
    gmsh.model.addPhysicalGroup(1, pin2_lines, 2)
    gmsh.model.setPhysicalName(1, 2, "pin_upper")

    # Refinement along crack path
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumber(1, "VOut", h_coarse)
    gmsh.model.mesh.field.setNumber(1, "XMin", V_TIP_X - 1)
    gmsh.model.mesh.field.setNumber(1, "XMax", W_CT)
    gmsh.model.mesh.field.setNumber(1, "YMin", Y_CENTER - 3)
    gmsh.model.mesh.field.setNumber(1, "YMax", Y_CENTER + 3)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(2)
    mf = joinpath(OUTPUT_DIR, "CT_mesh.msh"); gmsh.write(mf); gmsh.finalize()
    return mf
end

# ============================================================================
# Simulation driver
# ============================================================================
function run_simulation()
    mkpath(OUTPUT_DIR)
    println("\n" * "="^70)
    println("  CT specimen")
    println("="^70)

    mesh_file = create_CT_mesh()
    model = GmshDiscreteModel(mesh_file)
    Ω = Triangulation(model); dΩ = Measure(Ω, 2)

    Γ_upper = BoundaryTriangulation(model; tags="pin_upper")
    Γ_lower = BoundaryTriangulation(model; tags="pin_lower")
    dΓ_upper = Measure(Γ_upper, 2)
    dΓ_lower = Measure(Γ_lower, 2)

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    V_u = TestFESpace(model, reffe_u, conformity=:H1)
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
    load_steps   = range(F_MIN, F_MAX, length=n_half+1)
    unload_steps = range(F_MAX, F_MIN, length=n_half+1)[2:end]
    cycle_hist = Int[]; crack_hist = Float64[]
    ψ_t = zeros(nc); cur = 0

    t_upper(F) = VectorValue(0.0,  F)
    t_lower(F) = VectorValue(0.0, -F)

    println("  Starting fatigue cycling...")

    while cur < N_CYCLES_TOTAL
        cur += CYCLE_JUMP

        for (_, fr) in [("load", load_steps), ("unload", unload_steps)]
            for fv in fr
                g_s(s) = (1.0-s)^2 + η_stab

                a(u,v) = ∫((g_s ∘ sh) * (C_tensor ⊙ ε(u)) ⊙ ε(v)) * dΩ

                σp_vals = Vector{SymTensorValue{2,Float64,3}}(undef, nc)
                for i in 1:nc
                    σp_in = C_tensor ⊙ plastic_states[i].εp
                    add = mat.λ * plastic_states[i].εp33
                    σp_vals[i] = σp_in + SymTensorValue{2,Float64}(add, 0.0, add)
                end
                σp_f = CellField(σp_vals, Ω)

                l(v) = ∫((g_s ∘ sh) * σp_f ⊙ ε(v)) * dΩ +
                       ∫(t_upper(fv) ⋅ v) * dΓ_upper +
                       ∫(t_lower(fv) ⋅ v) * dΓ_lower

                op = AffineFEOperator(a, l, U_u, V_u)
                uh = solve(op)
                compute_stress!(uh, sh, plastic_states, mat, η_stab, Ω, cpc, true)

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
                    αi = gs*ψ_t[i]
                    ᾱ_acc[i] += CYCLE_JUMP * max(αi-α_prev[i],0.0)
                    α_prev[i] = αi
                end

                for _ in 1:max_stagger
                    so = copy(get_free_dof_values(sh))
                    Hf = project_to_fe_space(H_prev, V_L2, dΩ)
                    af = array_to_fe_field(ᾱ_acc, V_dg0, V_L2, dΩ)
                    sh = solve_phase_field(sh, Hf, af, V_s, U_s, dΩ;
                                           Gc=GcI, l0=l0, αT=αT,
                                           fatigue_func=fatigue_degradation_smooth)
                    norm(get_free_dof_values(sh)-so)/(norm(get_free_dof_values(sh))+1e-10) < stagger_tol && break
                end
            end
        end

        # Track crack tip
        x_tip = track_crack_tip_x(sh, Ω, V_TIP_X, Y_CENTER, l0)
        Δa = x_tip - V_TIP_X
        push!(cycle_hist, cur); push!(crack_hist, Δa)

        if cur % 1000 == 0 || cur <= CYCLE_JUMP
            max_s = maximum(get_free_dof_values(sh))
            @printf("  N=%6d  max(s)=%.4f  Δa=%.3f mm\n", cur, max_s, Δa)
        end

        cur % FORCE_GC_EVERY == 0 && GC.gc(false)
        x_tip > W_CT - 5.0 && break
    end

    writedlm(joinpath(OUTPUT_DIR, "data.csv"), hcat(cycle_hist, crack_hist), ',')
    println("  Done: $(cur) cycles, Δa=$(round(crack_hist[end], digits=3)) mm")
    return cycle_hist, crack_hist
end

function main()
    cycles, da = run_simulation()
    if !isempty(cycles)
        p = plot(cycles, da, xlabel="N", ylabel="Δa (mm)",
                 title="CT Crack Growth", label="Simulation",
                 linewidth=2, grid=true, size=(800,600), dpi=300)
        savefig(p, joinpath(OUTPUT_DIR, "crack_growth.png"))
    end
end

main()
