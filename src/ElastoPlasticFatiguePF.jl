module ElastoPlasticFatiguePF

using Gridap
using Gridap.Geometry
using Gridap.Geometry: get_node_coordinates, get_cell_coordinates
using Gridap.TensorValues
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Algebra
using GridapGmsh
using LinearAlgebra
using Statistics
using Printf
using Plots
using DelimitedFiles

const ε = symmetric_gradient

include("materials.jl")
include("plasticity.jl")
include("energy.jl")
include("phasefield.jl")
include("solver.jl")
include("projection.jl")
include("visualization.jl")

export MaterialData, elastic_stiffness_2D,
       yield_stress, yield_stress_derivative, plastic_strain_energy,
       PlasticStateData, copy_plastic_state,
       elastic_stress_components_plane_strain, von_mises_eq_3d,
       return_mapping!,
       tensile_elastic_strain_energy, tensile_elastic_strain_energy_components,
       principal_strains_2D, update_history_variable,
       g_degradation,
       fatigue_degradation_threshold, fatigue_degradation_smooth,
       solve_phase_field,
       compute_stress!, solve_displacement,
       project_to_fe_space, array_to_fe_field,
       QPScalarField, QPTensor2DField,
       build_qp_info, qp_scalar_cellfield, qp_tensor_cellfield, qp_to_cell_mean,
       plot_cell_scalar_png, plot_mesh_png, track_crack_tip_x,
       ε

end # module
