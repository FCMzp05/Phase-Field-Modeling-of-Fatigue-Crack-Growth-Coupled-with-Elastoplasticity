# Phase-field equation and fatigue degradation functions.
# AT-2 regularisation with threshold-based or smooth fatigue degradation.

# ---- degradation functions ----

@inline g_degradation(s::Real, η::Float64) = (1.0 - s)^2 + η

"""
    fatigue_degradation_threshold(ᾱ, αT) -> Float64

Threshold-based fatigue degradation (Eq. 4):
  f(ᾱ) = 1                   if ᾱ ≤ αT
  f(ᾱ) = (2αT / (ᾱ + αT))²  if ᾱ > αT
"""
@inline function fatigue_degradation_threshold(ᾱ::Real, αT::Real)
    return ᾱ <= αT ? 1.0 : (2.0 * αT / (ᾱ + αT))^2
end

"""
    fatigue_degradation_smooth(ᾱ, αT) -> Float64

Smooth fatigue degradation (no threshold):
  f(ᾱ) = (αT / (ᾱ + αT))²
"""
@inline function fatigue_degradation_smooth(ᾱ::Real, αT::Real)
    return (αT / (ᾱ + αT))^2
end

# ---- standard AT-2 phase-field solver ----

"""
    solve_phase_field(sh_prev, H_field, ᾱ_field, V_s, U_s, dΩ;
                      Gc, l0, αT, fatigue_func) -> FEFunction

Solve the AT-2 phase-field equation with irreversibility constraint.
"""
function solve_phase_field(
    sh_prev, H_field, ᾱ_field,
    V_s, U_s, dΩ;
    Gc::Float64,
    l0::Float64,
    αT::Float64,
    fatigue_func::Function = fatigue_degradation_threshold
)
    f_dg = (x -> fatigue_func(x, αT)) ∘ ᾱ_field
    P    = H_field

    a(s, ϕ) = ∫(
        (2.0 * P + (f_dg * Gc / l0)) * s * ϕ +
        f_dg * Gc * l0 * ∇(s) ⋅ ∇(ϕ)
    ) * dΩ

    b(ϕ) = ∫(2.0 * P * ϕ) * dΩ

    op = AffineFEOperator(a, b, U_s, V_s)
    sh_solved = solve(op)

    s_new = get_free_dof_values(sh_solved)
    s_old = get_free_dof_values(sh_prev)

    for i in eachindex(s_new)
        if isnan(s_new[i]) || isinf(s_new[i])
            s_new[i] = s_old[i]
        end
        s_new[i] = clamp(max(s_new[i], s_old[i]), 0.0, 1.0)
    end

    return FEFunction(U_s, s_new)
end
