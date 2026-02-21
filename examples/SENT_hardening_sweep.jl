#!/usr/bin/env julia
# ============================================================================
# SENT specimen -- hardening exponent parametric sweep
# Compares crack growth rates for different power-law exponents n.
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
const W = 1.0; const H = 1.0; const a0 = 0.5
const l0 = 0.02; const Gc = 2.7; const αT = 11.25
const η_stab = 1e-6; const η_plastic = 1.0
const u_max = 0.001; const R_load = 0.0; const u_min = 0.0
const N_CYCLES = 1500; const N_STEPS_PER_CYCLE = 20
const phase_every = 1; const max_stagger = 20; const stagger_tol = 1e-6
const MIN_CYCLES_BEFORE_STOP = 700; const STOP_CRACK_LENGTH = 50.0
const h_fine = 0.005; const h_coarse = 0.05; const FORCE_GC_EVERY = 20

const HARDENING_N_LIST = [0.0, 0.05, 0.1, 0.2, 0.3]
const OUTPUT_ROOT = "results_SENT_MULTIHARDENING"

# ============================================================================
# Mesh
# ============================================================================
function create_mesh(output_dir)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("SENT")
    y_crack = H/2; CH = h_fine
    gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_coarse, 1)
    gmsh.model.geo.addPoint(W, 0.0, 0.0, h_coarse, 2)
    gmsh.model.geo.addPoint(W, H/2, 0.0, h_coarse, 22)
    gmsh.model.geo.addPoint(W, H, 0.0, h_coarse, 3)
    gmsh.model.geo.addPoint(0.0, H, 0.0, h_coarse, 4)
    gmsh.model.geo.addPoint(0.0, y_crack+0.5*CH, 0.0, h_coarse, 5)
    gmsh.model.geo.addPoint(a0-CH, y_crack+0.5*CH, 0.0, h_fine, 6)
    gmsh.model.geo.addPoint(a0, y_crack, 0.0, h_fine, 7)
    gmsh.model.geo.addPoint(a0-CH, y_crack-0.5*CH, 0.0, h_fine, 8)
    gmsh.model.geo.addPoint(0.0, y_crack-0.5*CH, 0.0, h_coarse, 9)
    for (s,e,id) in [(1,2,1),(2,22,11),(22,3,2),(3,4,3),(4,5,4),
                      (5,6,5),(6,7,6),(7,8,7),(8,9,8),(9,1,9)]
        gmsh.model.geo.addLine(s, e, id)
    end
    gmsh.model.geo.addCurveLoop([1,11,2,3,4,5,6,7,8,9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1); gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2,[1],1); gmsh.model.setPhysicalName(2,1,"Domain")
    gmsh.model.addPhysicalGroup(1,[1],1); gmsh.model.setPhysicalName(1,1,"bottom")
    gmsh.model.addPhysicalGroup(1,[3],2); gmsh.model.setPhysicalName(1,2,"top")
    gmsh.model.mesh.field.add("Box",1)
    gmsh.model.mesh.field.setNumber(1,"VIn",h_fine)
    gmsh.model.mesh.field.setNumber(1,"VOut",h_coarse)
    gmsh.model.mesh.field.setNumber(1,"XMin",a0-0.1)
    gmsh.model.mesh.field.setNumber(1,"XMax",W)
    gmsh.model.mesh.field.setNumber(1,"YMin",H/2-0.2)
    gmsh.model.mesh.field.setNumber(1,"YMax",H/2+0.2)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.model.mesh.generate(2)
    mf = joinpath(output_dir, "mesh.msh"); gmsh.write(mf); gmsh.finalize()
    return mf
end

# ============================================================================
# Single-n simulation
# ============================================================================
function run_for_n(n_hard::Float64)
    outdir = joinpath(OUTPUT_ROOT, @sprintf("n_%.2f", n_hard))
    mkpath(outdir)
    println("\n--- n = $(n_hard) ---")

    mat_case = MaterialData(210000.0, 0.3, 440.0, n_hard)
    C_case   = elastic_stiffness_2D(mat_case)

    mesh_file = create_mesh(outdir)
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

    plastic_states = [PlasticStateData() for _ in 1:nc]
    H_prev = CellState(0.0, dΩ)
    ᾱ_acc = zeros(nc); α_prev = zeros(nc)
    sh = interpolate_everywhere(x->0.0, U_s); uh = zero(U_u)

    n_half = N_STEPS_PER_CYCLE ÷ 2
    load_steps   = range(u_min, u_max, length=n_half+1)
    unload_steps = range(u_max, u_min, length=n_half+1)[2:end]
    cycle_hist = Int[]; crack_hist = Float64[]
    u_prev_val = u_min; ψ_t = zeros(nc); cur = 0

    while cur < N_CYCLES
        cur += 1
        for (_, ur) in [("load", load_steps), ("unload", unload_steps)]
            for uv in ur
                U_u = TrialFESpace(V_u, [u_bottom, x->u_top(x,uv)])
                try
                    uh = solve_displacement(sh, plastic_states, uv,
                            V_u, U_u, dΩ, Ω, uh, u_prev_val,
                            cpc, mat_case, C_case, η_stab, u_max)
                catch; break; end
                ε_qp = evaluate(ε(uh), cpc); s_qp = evaluate(sh, cpc)
                for i in 1:nc
                    ψp = plastic_strain_energy(mat_case, plastic_states[i].εp_eq)
                    mx = 0.0
                    for q in 1:length(ε_qp[i])
                        mx = max(mx, tensile_elastic_strain_energy(
                                     ε_qp[i][q], plastic_states[i], mat_case))
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
        a_cur = track_crack_tip_x(sh, Ω, a0, H/2, l0)
        Δa = a_cur - a0
        push!(cycle_hist, cur); push!(crack_hist, Δa)
        cur % FORCE_GC_EVERY == 0 && GC.gc(false)
        cur >= MIN_CYCLES_BEFORE_STOP && Δa > STOP_CRACK_LENGTH && break
    end

    writedlm(joinpath(outdir, "data.csv"), hcat(cycle_hist, crack_hist), ',')
    @printf("  n=%.2f  N=%d  Δa=%.4f mm\n", n_hard, cur, crack_hist[end])
    return cur, crack_hist[end], outdir
end

# ============================================================================
# Batch entry point
# ============================================================================
function main()
    mkpath(OUTPUT_ROOT)
    println("SENT hardening exponent sweep")
    println("n list: ", HARDENING_N_LIST)
    summary = []
    for n_h in HARDENING_N_LIST
        Nf, da, od = run_for_n(n_h)
        push!(summary, (n_h, Nf, da, od))
    end
    open(joinpath(OUTPUT_ROOT, "summary.csv"), "w") do io
        println(io, "n_hard,N_final,delta_a_mm,output_dir")
        for (n, nf, d, o) in summary
            @printf(io, "%.4f,%d,%.6f,%s\n", n, nf, d, o)
        end
    end
    println("\nBatch done: ", joinpath(OUTPUT_ROOT, "summary.csv"))
end

main()
