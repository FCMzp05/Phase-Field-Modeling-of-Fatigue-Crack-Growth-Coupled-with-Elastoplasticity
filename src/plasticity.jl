# Plane-strain J2 radial return mapping in 3D effective stress space.
# The yield criterion is evaluated on the *undegraded* effective stress,
# and the degradation g(d) is applied only to the output nominal stress
# (Section 2.4 in the manuscript).

mutable struct PlasticStateData
    εp::SymTensorValue{2,Float64,3}   # in-plane plastic strain
    εp33::Float64                     # out-of-plane plastic strain (plane strain)
    εp_eq::Float64                    # accumulated equivalent plastic strain
end

PlasticStateData() = PlasticStateData(
    SymTensorValue{2,Float64}(0.0, 0.0, 0.0), 0.0, 0.0
)

copy_plastic_state(ps::PlasticStateData) =
    PlasticStateData(ps.εp, ps.εp33, ps.εp_eq)

# ---- 3D stress evaluation under plane strain ----

@inline function elastic_stress_components_plane_strain(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    mat::MaterialData
)
    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33
    trεe = εe11 + εe22 + εe33

    σ11 = 2.0 * mat.G * εe11 + mat.λ * trεe
    σ22 = 2.0 * mat.G * εe22 + mat.λ * trεe
    σ33 = 2.0 * mat.G * εe33 + mat.λ * trεe
    σ12 = 2.0 * mat.G * εe12

    return σ11, σ22, σ33, σ12
end

@inline function von_mises_eq_3d(σ11, σ22, σ33, σ12)
    p  = (σ11 + σ22 + σ33) / 3.0
    s11 = σ11 - p
    s22 = σ22 - p
    s33 = σ33 - p
    s12 = σ12
    σ_eq = sqrt(1.5 * (s11^2 + s22^2 + s33^2 + 2.0 * s12^2))
    return σ_eq, s11, s22, s33, s12
end

# ---- Radial return mapping (Appendix A, Eqs. 9--12) ----

"""
    return_mapping!(ε_total, plastic, mat, s_phasefield, η_stab;
                    update_state=true) -> SymTensorValue{2}

Plane-strain J2 radial return with 3D Von Mises.
Returns the *degraded* in-plane nominal stress tensor.
"""
function return_mapping!(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    mat::MaterialData,
    s_phasefield::Float64,
    η_stab::Float64;
    update_state::Bool = true
)
    σ11_tr, σ22_tr, σ33_tr, σ12_tr =
        elastic_stress_components_plane_strain(ε_total, plastic, mat)

    σ_eq_tr, s11, s22, s33, s12 =
        von_mises_eq_3d(σ11_tr, σ22_tr, σ33_tr, σ12_tr)

    σy = yield_stress(mat, plastic.εp_eq)
    g_s = (1.0 - s_phasefield)^2 + η_stab

    f_trial = σ_eq_tr - σy

    if f_trial <= 0.0 || σ_eq_tr < 1e-10
        return g_s * SymTensorValue{2,Float64}(σ11_tr, σ12_tr, σ22_tr)
    end

    Δγ  = 0.0
    tol = 1e-9 * mat.σy0

    for _ in 1:50
        εp_eq_new = plastic.εp_eq + Δγ
        σy_new    = yield_stress(mat, εp_eq_new)
        σ_eq_new  = σ_eq_tr - 3.0 * mat.G * Δγ

        residual = σ_eq_new - σy_new
        abs(residual) < tol && begin
            inv_seq = σ_eq_tr > 1e-12 ? (1.0 / σ_eq_tr) : 0.0
            Δεp11 = 1.5 * Δγ * s11 * inv_seq
            Δεp22 = 1.5 * Δγ * s22 * inv_seq
            Δεp33 = 1.5 * Δγ * s33 * inv_seq
            Δεp12 = 1.5 * Δγ * s12 * inv_seq

            if update_state
                plastic.εp    += SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22)
                plastic.εp33  += Δεp33
                plastic.εp_eq  = εp_eq_new
            end

            ps_tmp = update_state ? plastic : PlasticStateData(
                plastic.εp + SymTensorValue{2,Float64}(Δεp11, Δεp12, Δεp22),
                plastic.εp33 + Δεp33, εp_eq_new
            )
            σ11, σ22, _, σ12 =
                elastic_stress_components_plane_strain(ε_total, ps_tmp, mat)
            return g_s * SymTensorValue{2,Float64}(σ11, σ12, σ22)
        end

        dσy = yield_stress_derivative(mat, εp_eq_new)
        dR  = -3.0 * mat.G - dσy
        abs(dR) < 1e-15 && break
        Δγ = max(Δγ - residual / dR, 0.0)
    end

    @warn "Return mapping did not converge"
    return g_s * SymTensorValue{2,Float64}(σ11_tr, σ12_tr, σ22_tr)
end
