# Visualization utilities: cell-level PNG contour plots and mesh display.

"""
    plot_cell_scalar_png(cell_coords, cell_values; title, out_file, ...)

Render a per-cell scalar field as a filled contour PNG.
"""
function plot_cell_scalar_png(
    cell_coords, cell_values::Vector{Float64};
    title::String,
    out_file::String,
    dpi::Int = 120,
    cmap     = :jet,
    vmin::Union{Nothing,Float64} = nothing,
    vmax::Union{Nothing,Float64} = nothing
)
    vmin_use = isnothing(vmin) ? minimum(cell_values) : vmin
    vmax_use = isnothing(vmax) ? maximum(cell_values) : vmax
    if !(vmax_use > vmin_use)
        vmax_use = vmin_use + 1e-12
    end
    grad = cgrad(cmap)

    p = plot(aspect_ratio=:equal, size=(700, 700), dpi=dpi,
             xlabel="x (mm)", ylabel="y (mm)", title=title, legend=false)

    for i in eachindex(cell_coords)
        verts = cell_coords[i]
        xs = [v[1] for v in verts]
        ys = [v[2] for v in verts]
        val = isfinite(cell_values[i]) ? cell_values[i] : vmin_use
        ξ = clamp((val - vmin_use) / (vmax_use - vmin_use), 0.0, 1.0)
        plot!(p, Shape(xs, ys),
              fillcolor=grad[ξ], linecolor=:transparent,
              linewidth=0.0, label="")
    end

    savefig(p, out_file)
    closeall()
end

"""
    plot_mesh_png(model, out_file; title)

Save a mesh wireframe as PNG.
"""
function plot_mesh_png(model, out_file::String; title::String = "Mesh")
    try
        Ω = Triangulation(model)
        cell_coords = get_cell_coordinates(Ω)
        p = plot(aspect_ratio=:equal, size=(700, 700), dpi=300,
                 xlabel="x (mm)", ylabel="y (mm)", legend=false,
                 title="$(title) ($(num_cells(Ω)) elements)")
        for i_cell in 1:num_cells(Ω)
            verts = cell_coords[i_cell]
            xs = [v[1] for v in verts]
            ys = [v[2] for v in verts]
            plot!(p, Shape(xs, ys), fillcolor=:white,
                  linecolor=:black, linewidth=0.3)
        end
        savefig(p, out_file)
    catch e
        @warn "Mesh plotting failed: $e"
    end
end

"""
    track_crack_tip_x(sh, Ω, x0, y_crack, l0; tol=0.5)

Track the crack tip position along the x-axis (for horizontal cracks).
"""
function track_crack_tip_x(sh, Ω, x0::Float64, y_crack::Float64,
                           l0::Float64; tol::Float64 = 0.5)
    coords = get_node_coordinates(Ω)
    s_vals = get_free_dof_values(sh)
    x_max  = x0
    for i in eachindex(s_vals)
        x, y = coords[i][1], coords[i][2]
        if x > x0 - l0 && abs(y - y_crack) < 2.0 * l0 && s_vals[i] > tol
            x_max = max(x_max, x)
        end
    end
    return x_max
end
