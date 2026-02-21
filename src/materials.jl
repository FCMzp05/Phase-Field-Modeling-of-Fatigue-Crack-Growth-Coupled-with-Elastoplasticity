# Material data structure and hardening model for J2 elastoplasticity
# with power-law isotropic hardening (Eq. 13 in the manuscript).

struct MaterialData
    E::Float64       # Young's modulus [MPa]
    ν::Float64       # Poisson's ratio
    G::Float64       # Shear modulus [MPa]
    λ::Float64       # Lame's first parameter [MPa]
    σy0::Float64     # Initial yield stress [MPa]
    n::Float64       # Power-law hardening exponent
end

function MaterialData(E::Float64, ν::Float64, σy0::Float64, n::Float64)
    G = E / (2.0 * (1.0 + ν))
    λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))
    return MaterialData(E, ν, G, λ, σy0, n)
end

"""
    yield_stress(mat, εp_eq) -> Float64

Power-law isotropic hardening: σ_y = σ_{y0} (1 + E/σ_{y0} εp_eq)^n.
"""
@inline function yield_stress(mat::MaterialData, εp_eq::Float64)
    return mat.σy0 * (1.0 + (mat.E / mat.σy0) * εp_eq)^mat.n
end

"""
    yield_stress_derivative(mat, εp_eq) -> Float64

Derivative dσ_y/dεp_eq for Newton--Raphson in the return mapping.
"""
@inline function yield_stress_derivative(mat::MaterialData, εp_eq::Float64)
    term = 1.0 + (mat.E / mat.σy0) * εp_eq
    return term > 1e-10 ? mat.n * mat.E * term^(mat.n - 1.0) : 0.0
end

"""
    plastic_strain_energy(mat, εp_eq) -> Float64

Analytical integral ψ_p = ∫₀^{εp_eq} σ_y(α) dα  (Eq. 3).
"""
@inline function plastic_strain_energy(mat::MaterialData, εp_eq::Float64)
    abs(εp_eq) < 1e-12 && return 0.0
    if abs(mat.n + 1.0) < 1e-10
        return mat.σy0 * εp_eq
    end
    term = (1.0 + (mat.E / mat.σy0) * εp_eq)^(mat.n + 1.0)
    return (mat.σy0^2 / mat.E) * (term - 1.0) / (mat.n + 1.0)
end

"""
    elastic_stiffness_2D(mat) -> SymFourthOrderTensorValue

Plane-strain elastic stiffness tensor (2D Voigt).
"""
function elastic_stiffness_2D(mat::MaterialData)
    C1111 = mat.E * (1.0 - mat.ν) / ((1.0 + mat.ν) * (1.0 - 2.0 * mat.ν))
    C1122 = mat.E * mat.ν / ((1.0 + mat.ν) * (1.0 - 2.0 * mat.ν))
    C1212 = mat.E / (2.0 * (1.0 + mat.ν))
    return SymFourthOrderTensorValue(
        C1111, 0.0, C1122,
        0.0, C1212, 0.0,
        C1122, 0.0, C1111
    )
end
