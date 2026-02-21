# L2 projection utilities and custom quadrature-point field types.

"""
    project_to_fe_space(q, V_proj, dΩ) -> FEFunction

Standard L2 projection of a CellField / CellState onto `V_proj`.
"""
function project_to_fe_space(q, V_proj, dΩ)
    a(u, v) = ∫(u * v) * dΩ
    b(v)    = ∫(v * q) * dΩ
    op = AffineFEOperator(a, b, V_proj, V_proj)
    return solve(op)
end

"""
    array_to_fe_field(arr, V_dg0, V_proj, dΩ) -> FEFunction

Build a DG0 `FEFunction` from a raw array and project it to `V_proj`.
"""
function array_to_fe_field(arr, V_dg0, V_proj, dΩ)
    f_dg = FEFunction(V_dg0, arr)
    return project_to_fe_space(f_dg, V_proj, dΩ)
end

# ---- Custom QP-to-CellField types (used by QP-level state management) ----

struct QPScalarField <: Gridap.Fields.Field
    values::Vector{Float64}
end

function Gridap.Fields.return_cache(f::QPScalarField, x::AbstractVector{<:Point})
    return copy(f.values)
end

function Gridap.Fields.evaluate!(cache, f::QPScalarField, x::AbstractVector{<:Point})
    @assert length(x) == length(f.values)
    return f.values
end

struct QPTensor2DField <: Gridap.Fields.Field
    values::Vector{SymTensorValue{2,Float64,3}}
end

function Gridap.Fields.return_cache(f::QPTensor2DField, x::AbstractVector{<:Point})
    return copy(f.values)
end

function Gridap.Fields.evaluate!(cache, f::QPTensor2DField, x::AbstractVector{<:Point})
    @assert length(x) == length(f.values)
    return f.values
end

# ---- QP offset / cell-mean helpers ----

"""
    build_qp_info(Ω, degree) -> (offsets, nqp_cell, total_nqp)

Compute per-cell quadrature-point offsets for flat-array state storage.
"""
function build_qp_info(Ω, degree)
    quad   = CellQuadrature(Ω, degree)
    coords = collect(get_data(get_cell_points(quad)))
    nc     = length(coords)
    offsets  = zeros(Int, nc)
    nqp_cell = zeros(Int, nc)
    running  = 0
    for i in 1:nc
        offsets[i]  = running
        nqp_cell[i] = length(coords[i])
        running    += nqp_cell[i]
    end
    return offsets, nqp_cell, running
end

function qp_scalar_cellfield(data::Vector{Float64}, offsets, nqp_cell, Ω)
    nc   = length(offsets)
    vals = Vector{Float64}(undef, nc)
    for i in 1:nc
        o = offsets[i]; n = nqp_cell[i]
        acc = 0.0
        @inbounds for j in 1:n; acc += data[o + j]; end
        vals[i] = acc / n
    end
    return CellData.CellField(vals, Ω)
end

function qp_tensor_cellfield(d11, d12, d22, offsets, nqp_cell, Ω)
    nc   = length(offsets)
    vals = Vector{SymTensorValue{2,Float64,3}}(undef, nc)
    for i in 1:nc
        o = offsets[i]; n = nqp_cell[i]
        s11 = 0.0; s12 = 0.0; s22 = 0.0
        @inbounds for j in 1:n
            s11 += d11[o+j]; s12 += d12[o+j]; s22 += d22[o+j]
        end
        vals[i] = SymTensorValue{2,Float64,3}(s11/n, s12/n, s22/n)
    end
    return CellData.CellField(vals, Ω)
end

function qp_to_cell_mean(data::Vector{Float64}, offsets, nqp_cell)
    nc   = length(offsets)
    vals = zeros(Float64, nc)
    for i in 1:nc
        o = offsets[i]; n = nqp_cell[i]
        acc = 0.0
        @inbounds for j in 1:n; acc += data[o + j]; end
        vals[i] = acc / n
    end
    return vals
end
