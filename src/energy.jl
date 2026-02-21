# Strain energy decomposition and history variable update.
# Amor volumetric--deviatoric split (Eq. 2) is the default.

"""
    tensile_elastic_strain_energy(ε_total, plastic, mat) -> Float64

Amor volumetric--deviatoric split (3D plane strain):
  ψ_e⁺ = K/2 ⟨tr(ε_e)⟩₊² + G |dev(ε_e)|²
"""
function tensile_elastic_strain_energy(
    ε_total::SymTensorValue{2,Float64,3},
    plastic::PlasticStateData,
    mat::MaterialData
)
    εe11 = ε_total[1,1] - plastic.εp[1,1]
    εe22 = ε_total[2,2] - plastic.εp[2,2]
    εe12 = ε_total[1,2] - plastic.εp[1,2]
    εe33 = -plastic.εp33

    trεe   = εe11 + εe22 + εe33
    tr_pos = max(trεe, 0.0)
    K      = mat.λ + (2.0 / 3.0) * mat.G

    m     = trεe / 3.0
    dev11 = εe11 - m
    dev22 = εe22 - m
    dev33 = εe33 - m
    dev12 = εe12
    dev_norm2 = dev11^2 + dev22^2 + dev33^2 + 2.0 * dev12^2

    return 0.5 * K * tr_pos^2 + mat.G * dev_norm2
end

"""
    tensile_elastic_strain_energy_components(εe11, εe22, εe12, εe33, λ, G) -> Float64

Component-level interface used by the QP-based state management path.
"""
@inline function tensile_elastic_strain_energy_components(
    εe11, εe22, εe12, εe33, λ, G
)
    trεe   = εe11 + εe22 + εe33
    tr_pos = max(trεe, 0.0)
    K      = λ + 2.0 * G / 3.0
    m      = trεe / 3.0
    dev_norm2 = (εe11 - m)^2 + (εe22 - m)^2 + (εe33 - m)^2 + 2.0 * εe12^2
    return 0.5 * K * tr_pos^2 + G * dev_norm2
end

# ---- principal-strain helpers (for Miehe split) ----

function principal_strains_2D(ε::SymTensorValue{2,Float64,3})
    ε11, ε12, ε22 = ε[1,1], ε[1,2], ε[2,2]
    trace_ε = ε11 + ε22
    det_ε   = ε11 * ε22 - ε12^2
    disc    = sqrt(max(0.0, (trace_ε / 2.0)^2 - det_ε))
    return trace_ε / 2.0 + disc, trace_ε / 2.0 - disc
end

# ---- history variable ----

"""
    update_history_variable(ψ_total, H_old) -> (Bool, Float64)

Irreversibility enforcement: H = max(H_old, ψ_e⁺ + η ψ_p).
"""
@inline function update_history_variable(ψ_total::Real, H_old::Real)
    return (true, max(H_old, ψ_total))
end
