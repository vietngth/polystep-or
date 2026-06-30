"""
    Represents a single district in a districting problem.
"""

"""
    is_district_feasible(instance::Instance, district::District)::Bool

Determines if a given `district` is feasible within the specified `instance`.

# Arguments
- `instance::Instance`: The problem instance containing districting parameters.
- `district::District`: The district to evaluate.

# Returns
- `Bool`: `true` if the district meets the size constraints and is connected; `false` otherwise.
"""

function is_district_feasible(instance::Instance, district::District)
    n_nodes = length(district.nodes)
    return n_nodes >= instance.min_district_size &&
           n_nodes <= instance.max_district_size &&
           is_subgraph_connected(instance.graph, district.nodes)
end

"""
    compute_cost_with_precomputed_data(instance::Instance, district::District, costloader::Costloader)

Computes the cost of a district using precomputed data from `costloader`, falling back to `compute_cost_via_SAA` if necessary.

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district for which cost is being computed.
- `costloader::Costloader`: The precomputed cost data.

# Returns
- Cost of the district as per precomputed data or SSA computation.
"""

function compute_cost_with_precomputed_data(
    instance::Instance,
    district::District,
    costloader::Costloader,
)
    g = instance.graph
    nodes = district.nodes
    original_nodes = [get_prop(g, j, :id) for j in nodes]
    idx = check_1d_in_2d(original_nodes, costloader.DistrictIds)
    return idx === nothing ? compute_cost_via_SAA(instance, original_nodes) :
           costloader.Cost[idx]
end

"""
    compute_cost_via_SAA(instance::Instance, nodes::Array{Int})

Computes the cost of a district via Sample average approximation (SAA).

# Arguments
- `instance::Instance`: The problem instance.
- `nodes::Array{Int}`: Nodes representing the district.

# Returns
- The cost of the district as computed by SAA.
"""

function compute_cost_via_SAA(instance::Instance, nodes::Array{Int})
    g = instance.graph
    nodes = nodes .- 1
    formatted_nodes = "{" * join(nodes, ", ") * "}"
    target = string( instance.city_name, "_", instance.depot, "_", instance.num_blocks, "_", instance.target_district_size, "_", NB_SCENARIO)
    # The C++ SAA evaluator reads the scenario JSON; under concurrent scenario (re)generation across
    # parallel jobs a read can catch a truncated mid-write file and throw a JSON parse error. Retry with
    # backoff so a transient write/read race does not kill a multi-hour run; rethrow if it never settles.
    local err
    for attempt in 1:6
        try
            return EVmain(formatted_nodes, target)
        catch e
            err = e
            sleep(2.0 * attempt)
        end
    end
    rethrow(err)
end

"""
    compute_cost_via_CMST(instance::Instance, district::District)

Computes the cost of a district using the Capacitated Minimum Spanning Tree (CMST) approach.

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district to compute the cost for.

# Returns
- The computed cost based on CMST.
"""

function compute_cost_via_CMST(instance::Instance, district::District)
    nodes = district.nodes
    if length(nodes) <= 1
        return PENALITY * instance.max_district_size
    end
    sub_graph, _ = induced_subgraph(instance.graph, nodes)
    mst_edges = prim_mst(sub_graph)
    if length(mst_edges) < length(nodes) - 1
        # disconnected district: the selected BUs are not mutually reachable in the base graph,
        # so prim_mst yields a partial/empty forest (< n-1 edges). Such a district is infeasible;
        # penalize it like a singleton instead of summing over an empty edge set (which throws).
        return PENALITY * instance.max_district_size
    end
    cost = sum(get_prop(sub_graph, edge, :weight) for edge in mst_edges)
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALITY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost + compute_depot_cost(instance, nodes)
end

"""
    compute_depot_cost(instance::Instance, nodes::Array{Int})

Calculates the cost of a district based on the distance to a depot.

# Arguments
- `instance::Instance`: The problem instance.
- `nodes::Array{Int}`: The nodes representing the district.

# Returns
- The minimum cost of reaching the depot from any node in the district.
"""

function compute_depot_cost(instance::Instance, nodes::Array{Int})
    return minimum([
        get_prop(instance.graph, instance.num_blocks + 1, node, :weight) for node in nodes
    ])
end

"""
    compute_cost_via_model(instance::Instance, district::District, model)

Computes the cost of a district using a given model, such as a Graph Neural Network (GNN).

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district to compute the cost for.
- `model`: The model used for cost computation.

# Returns
- The cost of the district as determined by the model.
"""

function compute_cost_via_predGNN(instance::Instance, district::District, model_params)
    model, train_mean, train_std = model_params
    nodes = district.nodes
    if length(nodes) <= 1
        return PENALITY * instance.max_district_size
    end
    for i = 1:nv(instance.graph)
        set_prop!(instance.graph, i, :district, 0)
    end
    for i in nodes
        set_prop!(instance.graph, i, :district, 1)
    end
    gn = load_instance_feat(instance)
    gn.ndata.x = Float32.((gn.ndata.x .- train_mean) ./ train_std)
    LinearAlgebra.BLAS.set_num_threads(1)
    y_pred = model(gn, gn.ndata.x)
    cost = y_pred[1]
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALITY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost
end

"""
    compute_cost_via_BD(instance::Instance, district::District, params)

Computes the cost of a district using the BD approach.

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district to compute the cost for.
- `params`: Parameters necessary for the BD calculation.

# Returns
- The cost of the district as computed by BD.
"""

function compute_cost_via_BD(instance::Instance, district::District, params)
    nodes = district.nodes
    if length(nodes) <= 1
        return PENALITY * instance.max_district_size
    end
    blocks_scenarios, depot_corr, beta = params[1], params[2], params[3]
    total_area = get_total_area(instance, district.nodes)
    expected_clients = get_expected_number_of_clients(instance, district.nodes)
    scenarios = get_districts_scenarios(district.nodes, blocks_scenarios)
    avg_distance = MC_average_distance_to_depot(scenarios, depot_corr)
    cost = beta * sqrt(total_area * expected_clients) + 2 * avg_distance
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALITY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost
end

"""
    compute_cost_via_FIG(instance::Instance, district::District, params)

Computes the cost of a district using the FIG approach.

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district to compute the cost for.
- `params`: Parameters necessary for the FIG calculation.

# Returns
- The cost of the district as computed by FIG.
"""

function compute_cost_via_FIG(instance::Instance, district::District, params)
    nodes = district.nodes
    if length(nodes) <= 1
        return PENALITY * instance.max_district_size
    end
    blocks_scenarios, depot_corr, betas = params[1], params[2], params[3]
    Ad = get_total_area(instance, district.nodes)
    Rd = get_expected_number_of_clients(instance, district.nodes)
    scenarios = get_districts_scenarios(district.nodes, blocks_scenarios)
    delta_d = MC_average_distance_to_depot(scenarios, depot_corr)
    #chech for NAN values and replace them with 1000
    if isnan(delta_d)
        delta_d = 1000
    end
    if isnan(Ad)
        Ad = 1000
    end
    if isnan(Rd)
        Rd = 1000
    end
    cost =
        betas[1] * sqrt(Ad * Rd) + betas[2] * delta_d + betas[3] * sqrt(Ad / Rd) + betas[4]
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALITY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost
end


function compute_cost_via_AvgTSP(instance::Instance, district::District)
    nodes = district.nodes
    g = instance.graph
    nodes = [get_prop(g, j, :id) for j in nodes]
    if length(nodes) <= 1
        return PENALITY * instance.max_district_size
    end
    cost = compute_cost_via_SAA(instance, nodes)
    ls, us = instance.min_district_size, instance.max_district_size
    cost += PENALITY * max(0, length(district.nodes) - us, ls - length(district.nodes))
    return cost
end

"""
    compute_district_cost(instance::Instance, district::District, costloader::Costloader, mode::String="CMST", model = nothing)

Computes the cost of a district based on the specified computation mode.

# Arguments
- `instance::Instance`: The problem instance.
- `district::District`: The district for which the cost is computed.
- `costloader::Costloader`: Precomputed cost data loader.
- `mode::String`: The mode of cost computation (e.g., "CMST", "GNN").
- `model`: The model used for cost computation, applicable in certain modes.

# Returns
- The computed cost of the district.
"""

function compute_district_cost(
    instance::Instance,
    district::District,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)
    if mode == "Districting"
        return compute_cost_with_precomputed_data(instance, district, costloader)
    elseif mode == "CMST"
        return compute_cost_via_CMST(instance, district)
    elseif mode == "GNN"
        return compute_cost_via_predGNN(instance, district, model)
    elseif mode == "BD"
        return compute_cost_via_BD(instance, district, model)
    elseif mode == "FIG"
        return compute_cost_via_FIG(instance, district, model)
    elseif mode == "AvgTSP"
        return compute_cost_via_AvgTSP(instance, district)
    else
        error("Invalid mode")
    end
end

"""
    create_district(i::Int, nodes::Vector{Int64}, instance::Instance, costloader::Costloader, mode::String="CMST", model = nothing)::District

Creates a district with a computed cost and feasibility status.

# Arguments
- `i::Int`: The identifier of the district.
- `nodes::Vector{Int64}`: Nodes representing the district.
- `instance::Instance`: The problem instance.
- `costloader::Costloader`: Precomputed cost data loader.
- `mode::String`: The mode of cost computation.
- `model`: The model used for cost computation, if applicable.

# Returns
- `District`: The created district with cost and feasibility information.
"""

function create_district(
    i::Int,
    nodes::Vector{Int64},
    instance::Instance,
    costloader::Costloader,
    mode::String = "CMST",
    model = nothing,
)::District
    cost = compute_district_cost(
        instance,
        District(i, nodes, 0.0, false),
        costloader,
        mode,
        model,
    )
    feasible = is_district_feasible(instance, District(i, nodes, cost, false))
    return District(i, nodes, cost, feasible)
end
