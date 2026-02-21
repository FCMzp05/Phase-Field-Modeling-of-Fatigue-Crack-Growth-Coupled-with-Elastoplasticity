#!/usr/bin/env julia
# ============================================================================
# Perforated (hole) plate fatigue specimen -- quarter-symmetric model
# Reproduces Section 4.1.1 of the manuscript.
# Two sub-cases: force-controlled and displacement-controlled.
# ============================================================================

using Gmsh: gmsh
using Gridap, Gridap.Geometry, Gridap.TensorValues, Gridap.CellData
using Gridap.FESpaces, Gridap.Algebra
using Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using GridapGmsh, LinearAlgebra, Statistics, Printf, Plots; gr(show=false)
using DelimitedFiles, WriteVTK, SparseArrays
ENV["GKSwstype"] = "100"

include(joinpath(@__DIR__, "..", "src", "ElastoPlasticFatiguePF.jl"))
using .ElastoPlasticFatiguePF

# ============================================================================
# Parameters
# ============================================================================
const E_mod  = 213.0e3;  const ν_val = 0.33
const λ_lame = E_mod*ν_val/((1+ν_val)*(1-2ν_val))
const μ_lame = E_mod/(2*(1+ν_val))
const σ_y0 = 100.0; const n_hard = 0.1
const mat = MaterialData(E_mod, ν_val, σ_y0, n_hard)

const l_0 = 0.3; const Gc_val = 54.0; const η_stab = 1e-10
const α_T = 100.0; const η_plastic = 1.0

const L_half = 30.0; const R_hole = 6.0
const h_fine = 0.08; const h_coarse = 1.2; const refine_dist = 16.0
const n_steps_half = 20
const f_max = 125.0; const f_min = -25.0
const u_max_load = 0.018; const u_min_load = -0.006
const n_cycles_max = 400; const cycle_jump = 10
const tol_am = 1e-4; const max_am_iter = 100
const max_picard = 30; const tol_picard = 1e-5
const damage_stop = 0.95; const quad_degree = 4
const OUTPUT_BASE = "results_hole_plate"

# ============================================================================
# QP-level state (flat arrays for this benchmark)
# ============================================================================
mutable struct HoleState
    εp11::Vector{Float64}; εp22::Vector{Float64}
    εp12::Vector{Float64}; εp33::Vector{Float64}; εp_eq::Vector{Float64}
    HI::Vector{Float64}; ψe::Vector{Float64}; ψp::Vector{Float64}
    ψd_old::Vector{Float64}; α_bar::Vector{Float64}
    σp11::Vector{Float64}; σp22::Vector{Float64}; σp12::Vector{Float64}
    nqp::Int
end
HoleState(n) = HoleState([zeros(n) for _ in 1:13]..., n)

function compute_σp!(st::HoleState)
    @inbounds for i in 1:st.nqp
        tr_ep = st.εp11[i]+st.εp22[i]+st.εp33[i]
        st.σp11[i] = λ_lame*tr_ep + 2μ_lame*st.εp11[i]
        st.σp22[i] = λ_lame*tr_ep + 2μ_lame*st.εp22[i]
        st.σp12[i] = 2μ_lame*st.εp12[i]
    end
end

# ============================================================================
# Local return mapping (component-level)
# ============================================================================
function rm_local(e11,e22,e12, ep11,ep22,ep12,ep33,ep_eq)
    ee11=e11-ep11; ee22=e22-ep22; ee12=e12-ep12; ee33=-ep33
    tr=ee11+ee22+ee33
    s11_tr=λ_lame*tr+2μ_lame*ee11; s22_tr=λ_lame*tr+2μ_lame*ee22
    s33_tr=λ_lame*tr+2μ_lame*ee33; s12_tr=2μ_lame*ee12
    σm=(s11_tr+s22_tr+s33_tr)/3
    d11=s11_tr-σm; d22=s22_tr-σm; d33=s33_tr-σm; d12=s12_tr
    J2=0.5*(d11^2+d22^2+d33^2)+d12^2; σeq=sqrt(3J2)
    σy_c=yield_stress(mat, ep_eq)
    if σeq <= σy_c + 1e-10
        ψp_v=plastic_strain_energy(mat, ep_eq)
        ψe_v=tensile_elastic_strain_energy_components(ee11,ee22,ee12,ee33,λ_lame,μ_lame)
        return (s11_tr,s22_tr,s12_tr,s33_tr, ep11,ep22,ep12,ep33,ep_eq, ψp_v,ψe_v)
    end
    Δγ=0.0; epk=ep_eq
    for _ in 1:50
        σy_k=yield_stress(mat,epk); dσy=yield_stress_derivative(mat,epk)
        fv=(σeq-3μ_lame*Δγ)-σy_k; abs(fv)<1e-10 && break
        Δγ -= fv/(-3μ_lame-dσy); Δγ=max(Δγ,0.0); epk=ep_eq+Δγ
    end
    r=σeq>1e-14 ? 3μ_lame*Δγ/σeq : 0.0
    fac=σeq>1e-14 ? 1.5Δγ/σeq : 0.0
    ep11_n=ep11+fac*d11; ep22_n=ep22+fac*d22; ep33_n=ep33+fac*d33; ep12_n=ep12+fac*d12
    ee11_n=e11-ep11_n; ee22_n=e22-ep22_n; ee12_n=e12-ep12_n; ee33_n=-ep33_n
    ψe_v=tensile_elastic_strain_energy_components(ee11_n,ee22_n,ee12_n,ee33_n,λ_lame,μ_lame)
    ψp_v=plastic_strain_energy(mat, ep_eq+Δγ)
    return (s11_tr-r*d11, s22_tr-r*d22, s12_tr-r*d12, s33_tr-r*d33,
            ep11_n,ep22_n,ep12_n,ep33_n, ep_eq+Δγ, ψp_v, ψe_v)
end

# ============================================================================
# Mesh
# ============================================================================
function create_mesh(msh_file)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("perf")
    p1=gmsh.model.geo.addPoint(R_hole,0,0,h_fine)
    p2=gmsh.model.geo.addPoint(L_half,0,0,h_coarse)
    p3=gmsh.model.geo.addPoint(L_half,L_half,0,h_coarse)
    p4=gmsh.model.geo.addPoint(0,L_half,0,h_coarse)
    p5=gmsh.model.geo.addPoint(0,R_hole,0,h_fine)
    p0=gmsh.model.geo.addPoint(0,0,0,h_fine)
    lb=gmsh.model.geo.addLine(p1,p2)
    lr=gmsh.model.geo.addLine(p2,p3)
    lt=gmsh.model.geo.addLine(p3,p4)
    ll=gmsh.model.geo.addLine(p4,p5)
    lh=gmsh.model.geo.addCircleArc(p5,p0,p1)
    cl=gmsh.model.geo.addCurveLoop([lb,lr,lt,ll,lh])
    sf=gmsh.model.geo.addPlaneSurface([cl]); gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2,[sf],100); gmsh.model.setPhysicalName(2,100,"domain")
    gmsh.model.addPhysicalGroup(1,[lb],1); gmsh.model.setPhysicalName(1,1,"bottom")
    gmsh.model.addPhysicalGroup(1,[ll],2); gmsh.model.setPhysicalName(1,2,"left")
    gmsh.model.addPhysicalGroup(1,[lt],3); gmsh.model.setPhysicalName(1,3,"top")
    gmsh.model.mesh.field.add("Distance",1)
    gmsh.model.mesh.field.setNumbers(1,"CurvesList",[lh])
    gmsh.model.mesh.field.setNumber(1,"Sampling",200)
    gmsh.model.mesh.field.add("Threshold",2)
    gmsh.model.mesh.field.setNumber(2,"InField",1)
    gmsh.model.mesh.field.setNumber(2,"SizeMin",h_fine)
    gmsh.model.mesh.field.setNumber(2,"SizeMax",h_coarse)
    gmsh.model.mesh.field.setNumber(2,"DistMin",0.0)
    gmsh.model.mesh.field.setNumber(2,"DistMax",refine_dist)
    gmsh.model.mesh.field.add("Min",3)
    gmsh.model.mesh.field.setNumbers(3,"FieldsList",[2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)
    gmsh.option.setNumber("Mesh.Algorithm",6)
    gmsh.model.mesh.generate(2); gmsh.write(msh_file); gmsh.finalize()
end

# ============================================================================
# QP-level updates
# ============================================================================
function update_plasticity!(st, offsets, nqp_cell, uh, s_field, Ω, degree)
    εh=ε(uh); quad=CellQuadrature(Ω,degree); qc=get_cell_points(quad)
    ε_v=collect(εh(qc)); s_v=collect(s_field(qc))
    for ic in 1:length(ε_v)
        for iq in 1:nqp_cell[ic]
            gid=offsets[ic]+iq; eq=ε_v[ic][iq]
            (_,_,_,_, ep11,ep22,ep12,ep33,epeq, ψp,ψe) =
                rm_local(eq[1,1],eq[2,2],eq[1,2],
                         st.εp11[gid],st.εp22[gid],st.εp12[gid],
                         st.εp33[gid],st.εp_eq[gid])
            st.εp11[gid]=ep11; st.εp22[gid]=ep22
            st.εp12[gid]=ep12; st.εp33[gid]=ep33; st.εp_eq[gid]=epeq
            st.ψe[gid]=ψe; st.ψp[gid]=ψp
            st.HI[gid]=max(st.HI[gid], ψe+η_plastic*ψp)
        end
    end
    compute_σp!(st)
end

function update_fatigue!(st, offsets, nqp_cell, s_field, Ω, degree, njump)
    quad=CellQuadrature(Ω,degree); qc=get_cell_points(quad)
    s_v=collect(s_field(qc))
    for ic in 1:length(s_v)
        for iq in 1:nqp_cell[ic]
            gid=offsets[ic]+iq
            s_qp = s_v[ic] isa Number ? s_v[ic] : s_v[ic][iq]
            gs=g_degradation(s_qp, η_stab)
            ψtot=st.ψe[gid]+η_plastic*st.ψp[gid]
            ψd=gs*ψtot
            Δψ=max(0.0, ψd-st.ψd_old[gid])
            st.α_bar[gid] += njump*Δψ
            st.ψd_old[gid]=ψd
        end
    end
end

# ============================================================================
# Solvers (QP-level)
# ============================================================================
function solve_u_hole!(uh, model, Ω, degree, st, offsets, nqp_cell, s_field,
                       load_val, is_force)
    dΩ=Measure(Ω,degree)
    Γ_top=BoundaryTriangulation(model; tags="top"); dΓ=Measure(Γ_top,degree)
    reffe_u=ReferenceFE(lagrangian, VectorValue{2,Float64}, 1)
    if is_force
        V=TestFESpace(model,reffe_u;conformity=:H1,
            dirichlet_tags=["bottom","left"],dirichlet_masks=[(false,true),(true,false)])
        U=TrialFESpace(V,[VectorValue(0,0),VectorValue(0,0)])
        t_top=VectorValue(0.0,load_val)
    else
        V=TestFESpace(model,reffe_u;conformity=:H1,
            dirichlet_tags=["bottom","left","top"],
            dirichlet_masks=[(false,true),(true,false),(false,true)])
        U=TrialFESpace(V,[VectorValue(0,0),VectorValue(0,0),VectorValue(0,load_val)])
        t_top=VectorValue(0,0)
    end
    gs_cf=g_degradation.(s_field, η_stab)
    a(u,v)=∫(gs_cf*(λ_lame*tr(ε(u))*tr(ε(v))+2μ_lame*(ε(u)⊙ε(v))))dΩ
    uh_c=uh
    for _ in 1:max_picard
        σp_cf=qp_tensor_cellfield(st.σp11,st.σp12,st.σp22,offsets,nqp_cell,Ω)
        l = if is_force
            v -> ∫(gs_cf*(σp_cf⊙ε(v)))dΩ + ∫(t_top⋅v)dΓ
        else
            v -> ∫(gs_cf*(σp_cf⊙ε(v)))dΩ
        end
        op=AffineFEOperator(a,l,U,V); uh_new=solve(op)
        update_plasticity!(st,offsets,nqp_cell,uh_new,s_field,Ω,degree)
        eh=uh_new-uh_c
        err=sqrt(abs(sum(∫(eh⋅eh)dΩ)))/(sqrt(abs(sum(∫(uh_new⋅uh_new)dΩ)))+1e-14)
        uh_c=uh_new; err<tol_picard && break
    end
    return uh_c
end

function solve_s_hole!(sh, model, Ω, degree, st, offsets, nqp_cell)
    dΩ=Measure(Ω,degree)
    V=TestFESpace(model, ReferenceFE(lagrangian,Float64,1); conformity=:H1)
    S=TrialFESpace(V)
    HI_cf=qp_scalar_cellfield(st.HI,offsets,nqp_cell,Ω)
    αbar_cf=qp_scalar_cellfield(st.α_bar,offsets,nqp_cell,Ω)
    f_cf=(x->fatigue_degradation_threshold(x,α_T)) ∘ αbar_cf
    a_s(s,δs)=∫((2.0*HI_cf+f_cf*Gc_val/l_0)*s*δs+(f_cf*Gc_val*l_0)*(∇(s)⋅∇(δs)))dΩ
    l_s(δs)=∫(2.0*HI_cf*δs)dΩ
    op=AffineFEOperator(a_s,l_s,S,V); sh_new=solve(op)
    sv=get_free_dof_values(sh_new); sv_old=get_free_dof_values(sh)
    for i in eachindex(sv)
        sv[i]=clamp(sv[i],0.0,1.0); sv[i]=max(sv[i],sv_old[i])
    end
    return FEFunction(S, sv)
end

# ============================================================================
# Simulation driver
# ============================================================================
function run_simulation(case_name, is_force)
    outdir = joinpath(OUTPUT_BASE, case_name); mkpath(outdir)
    println("\n" * "="^60)
    println("  HOLE PLATE: $(uppercase(case_name)) control")
    println("="^60)

    msh_file=joinpath(outdir,"mesh.msh"); create_mesh(msh_file)
    model=GmshDiscreteModel(msh_file); Ω=Triangulation(model)
    offsets,nqp_cell,total_nqp=build_qp_info(Ω,quad_degree)
    println("  Elements: $(num_cells(Ω)), QPs: $(total_nqp)")

    st=HoleState(total_nqp)
    V_s=TestFESpace(model, ReferenceFE(lagrangian,Float64,1); conformity=:H1)
    S_space=TrialFESpace(V_s)
    sh=FEFunction(S_space, zeros(num_free_dofs(S_space)))

    reffe_u=ReferenceFE(lagrangian,VectorValue{2,Float64},1)
    if is_force
        V0=TestFESpace(model,reffe_u;conformity=:H1,
            dirichlet_tags=["bottom","left"],dirichlet_masks=[(false,true),(true,false)])
        U0=TrialFESpace(V0,[VectorValue(0,0),VectorValue(0,0)])
    else
        V0=TestFESpace(model,reffe_u;conformity=:H1,
            dirichlet_tags=["bottom","left","top"],
            dirichlet_masks=[(false,true),(true,false),(false,true)])
        U0=TrialFESpace(V0,[VectorValue(0,0),VectorValue(0,0),VectorValue(0,0)])
    end
    uh=FEFunction(U0, zeros(num_free_dofs(U0)))

    lmin = is_force ? f_min : u_min_load
    lmax = is_force ? f_max : u_max_load

    for cycle in 1:n_cycles_max
        t0=time()
        for step in 1:n_steps_half
            lv=lmin+step/n_steps_half*(lmax-lmin)
            for _ in 1:max_am_iter
                sv_old=copy(get_free_dof_values(sh))
                uh=solve_u_hole!(uh,model,Ω,quad_degree,st,offsets,nqp_cell,sh,lv,is_force)
                sh=solve_s_hole!(sh,model,Ω,quad_degree,st,offsets,nqp_cell)
                am_err=norm(get_free_dof_values(sh).-sv_old)/(norm(get_free_dof_values(sh))+1e-14)
                am_err<tol_am && break
            end
        end
        for step in 1:n_steps_half
            lv=lmax-step/n_steps_half*(lmax-lmin)
            for _ in 1:max_am_iter
                sv_old=copy(get_free_dof_values(sh))
                uh=solve_u_hole!(uh,model,Ω,quad_degree,st,offsets,nqp_cell,sh,lv,is_force)
                sh=solve_s_hole!(sh,model,Ω,quad_degree,st,offsets,nqp_cell)
                am_err=norm(get_free_dof_values(sh).-sv_old)/(norm(get_free_dof_values(sh))+1e-14)
                am_err<tol_am && break
            end
        end
        njump = cycle <= 5 ? 1 : cycle_jump
        update_fatigue!(st,offsets,nqp_cell,sh,Ω,quad_degree,njump)
        max_s=maximum(get_free_dof_values(sh))
        @printf("  Cycle %4d  max(s)=%.4f  (%.1fs)\n", cycle, max_s, time()-t0)
    end
    println("  Done: $(outdir)/")
end

function main()
    println("Perforated specimen fatigue simulation")
    run_simulation("force", true)
    run_simulation("disp", false)
end

main()
