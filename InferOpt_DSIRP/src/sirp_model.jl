mutable struct Vertex
    id::Int
    x::Int
    y::Int

    Vertex() = new(0, 0, 0)
    Vertex(id, x, y) = new(id, x, y)
end


mutable struct IRPInstance
    name::String
    n::Int # Total number of customers, including depot
    K::Int # Total number of vehicles available
    v_cap::Int # Transportation capacity for each vehicle (homogenous)
    max_inventory::Dict{Int, Int} # Maximum allowed inventory for each customer
    start_inventory::Dict{Int, Float64}
    
    nb_features::Int
    
    demands_hist::Dict{Int, Vector{Float64}} # Demands of each customer per instant
    demands_eval::Dict{Int, Dict{Int, Vector{Float64}}} # Demands of each customer per instant
    demands_test::Dict{Int, Vector{Float64}} # Demands of each customer per instant
    
    samples_hist::Dict{Int, Vector{Any}} # Demands of each customer per instant
    samples_eval::Dict{Int, Dict{Int, Vector{Any}}} # Demands of each customer per instant
    samples_test::Dict{Int, Vector{Any}} # Demands of each customer per instant
    sample_demands::Dict{Int, Any} # Demands of each customer per instant (function to sample)

    holding_cost::Dict{Int, Float64} # Unit holding cost at each customer
    penalty_cost::Dict{Int, Float64} # Unit penalty cost at each customer
    vertices::Dict{Int, Vertex}
    distances::Dict{Int, Dict{Int, Float64}}

    IRPInstance() = new("", 0, 0, 0, Dict{Int, Int}(), Dict{Int, Float64}(), 0,
            Dict{Int, AbstractVector}(), Dict{Int, Dict{Int, AbstractVector}}(), Dict{Int, AbstractVector}(),
            Dict{Int, Any}(), Dict{Int, Dict{Int, Any}}(), Dict{Int, Any}(),
            Dict{Int, Float64}(), Dict{Int, Float64}(), Dict{Int, Vertex}(), Dict{Int, Dict{Int, Float64}}())
end


mutable struct IRPProblem
    n::Int # Total number of customers, including depot
    K::Int # Total number of vehicles available
    v_cap::Int # Transportation capacity for each vehicle (homogenous)
    max_inventory::Dict{Int, Int} # Maximum allowed inventory for each customer
    start_inventory::Dict{Int, Float64}
    horizon::Int # Number of discrete time instants of the planning horizon
    scenarios::Int
    demands::Dict{Int, Dict{Int, Vector{Float64}}}
    holding_cost::Dict{Int, Float64} # Unit holding cost at each customer
    penalty_cost::Dict{Int, Float64} # Unit penalty cost at each customer
    distances::Dict{Int, Dict{Int, Float64}}
    
    OU_policy::Bool

    IRPProblem() = new(0, 0, 0, Dict{Int, Int}(), Dict{Int, Float64}(), 0, 0, Dict{Int, Dict{Int, AbstractVector}}(), Dict{Int, Float64}(), Dict{Int, Float64}(), Dict{Int, Dict{Int, Float64}}(), true)
end


function createProblem(problem::IRPProblem, instance::IRPInstance; horizon=6, scenarios=1)
    
    problem.n = instance.n
    problem.K = instance.K
    problem.v_cap = instance.v_cap
    problem.max_inventory = instance.max_inventory
    problem.start_inventory = deepcopy(instance.start_inventory)
    problem.horizon = horizon 
    problem.scenarios = scenarios
    problem.holding_cost = instance.holding_cost
    problem.penalty_cost = instance.penalty_cost
    problem.distances = instance.distances
    
end


mutable struct IRPIndices
      V::Array{Int} # initial customer indices including depot
      V_cus::Array{Int} # only customers set
      V_aug::Array{Int} # customers, depot and copy depot for TCF formulation
      K::Array{Int} # vehicle indices
      T::Array{Int} # all time periods indices, including 0, for initial inventory
      H::Array{Int} # time periods in planning horizon
      S::Array{Int} # number of demand scenarios

      IRPIndices() = new(Int[], Int[], Int[], Int[], Int[], Int[], Int[])
end


function createIndices(indices::IRPIndices, problem::IRPProblem)
  indices.V = collect(1:problem.n)
  indices.V_cus = collect(2:problem.n)
  indices.V_aug = collect(1:(problem.n+1))
  indices.K = collect(1:problem.K)
  indices.T = collect(0:problem.horizon)
  indices.H = collect(1:problem.horizon)
  indices.S = collect(1:problem.scenarios)
end


mutable struct IRPSample
    n::Int
    distances::Dict{Int, Dict{Int, Float64}}
    holding_cost::Dict{Int, Float64} # Unit holding cost at each customer
    penalty_cost::Dict{Int, Float64} # Unit penalty cost at each customer
    max_inventory::Dict{Int, Int}
    start_inventory::Dict{Int, Float64}
    demands_hist::Dict{Int, Vector{Float64}}
    v_cap::Int
    feature_array::Array{Float64, 3}
    label::Array{Int}
    
    IRPSample() = new(0, Dict{Int, Float64}(), Dict{Int, Float64}(), Dict{Int, Dict{Int, Float64}}(), Dict{Int, Int}(), Dict{Int, Float64}(), Dict{Int, Vector{Float64}}(), 0, zeros(0,0,0), Int[])
end



function createFeatureArray(instance::IRPInstance, indices::IRPIndices, start_inventory::Dict{Int, Float64}, demands_hist::Dict{Int, Vector{Float64}}; contextual_features=Dict{Int, Dict{Int, Vector{Float64}}}(), look_ahead=6, demand_quantiles=[0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99], nb_features=0)
    if contextual_features == Dict{Int, Dict{Int, Vector{Float64}}}()
        return cat([reduce(hcat, [reduce(vcat, [start_inventory[i], q, instance.holding_cost[i], instance.penalty_cost[i]])
                    for q in quantile(demands_hist[i], demand_quantiles) for k in 1:look_ahead]) for i in indices.V_cus]..., dims=3)
    else
        return cat([reduce(hcat, [reduce(vcat, [start_inventory[i], q, instance.holding_cost[i], instance.penalty_cost[i], cat([contextual_features[k][i], [(contextual_features[k][i][n] * contextual_features[k][i][m]) for n in 1:nb_features for m in n+1:nb_features]]...,dims=1)])
                    for q in quantile(demands_hist[i], demand_quantiles) for k in 1:look_ahead]) for i in indices.V_cus]..., dims=3)
    end
end


function createSample(instance::IRPInstance, indices::IRPIndices, start_inventory::Dict{Int, Float64}, demands_hist::Dict{Int, Vector{Float64}}, label::Array{Int}; kwargs...)
    
    sample = IRPSample()
    sample.n = deepcopy(instance.n)
    sample.distances = deepcopy(instance.distances)
    sample.holding_cost = deepcopy(instance.holding_cost)
    sample.penalty_cost = deepcopy(instance.penalty_cost)
    sample.max_inventory = deepcopy(instance.max_inventory)
    sample.start_inventory = deepcopy(start_inventory)
    sample.demands_hist = deepcopy(demands_hist)
    sample.v_cap = deepcopy(instance.v_cap)
    
    sample.feature_array = deepcopy(createFeatureArray(instance, indices, start_inventory, demands_hist; kwargs...))
    sample.label = deepcopy(label)
    
    return sample
end

function computeDistances(vertices::Dict{Int, Vertex})
    distances = Dict{Int, Dict{Int, Float64}}()
    for (k1, c1) in vertices
        distances[c1.id] = Dict{Int, Float64}()
        for (k2, c2) in vertices
            dist = 0.0
            if c1.id != c2.id
                dist = sqrt(((c1.x - c2.x)^2) + ((c1.y - c2.y)^2))
            end
            distances[c1.id][c2.id] = dist
        end
    end
    return distances
end


function readInstance(path::String, pattern::String, instance::IRPInstance; penalty_inv=200., K=1)
    
    data = JSON.parsefile(realpath(path))
    vehicle_capacity = round((data["vehicle_capacity"] / K), RoundNearestTiesUp)

    instance.name = data["idx"]
    instance.n = data["nb_cst"] + 1
    instance.K = K    
    instance.v_cap = vehicle_capacity
    instance.nb_features = data["nb_features"]

    instance.vertices[1] = Vertex(1, 0, 0)

    for id in 1:(data["nb_cst"])
        _id = id + 1
        instance.max_inventory[_id] = data["max_inventory"][id]
        if pattern=="bimodal"
            instance.start_inventory[_id] = data["max_inventory"][id] - mean(data["mean_demand"][id])
        elseif pattern=="uniform"
            # uniform-10 instances ship no mean_demand; the authors' demand is Uniform(0, 0.5*max)
            # (see sample_demands below), whose mean is 0.25*max, so the (start = max - mean) rule gives 0.75*max.
            instance.start_inventory[_id] = 0.75 * data["max_inventory"][id]
        else
            instance.start_inventory[_id] = data["max_inventory"][id] - data["mean_demand"][id]
        end
        
        if pattern == "contextual"
            instance.samples_hist[_id] = collect(values(data["samples_hist"][string(id)]))
            instance.samples_eval[_id] = Dict{Int, Vector{Any}}()
            instance.demands_eval[_id] = Dict{Int, Vector{Float64}}()
            for (scenario, (_, demand_scenario)) in enumerate(data["samples_eval"][string(id)])
                instance.samples_eval[_id][scenario] = collect(values(demand_scenario))
                instance.demands_eval[_id][scenario] = FloatFromAny([x["label"] for x in collect(values(demand_scenario))])
            end
            instance.samples_test[_id] = collect(values(data["samples_test"][string(id)]))
            
            instance.demands_hist[_id] = FloatFromAny([x["label"] for x in values(data["samples_hist"][string(id)])])
            instance.demands_test[_id] = FloatFromAny([x["label"] for x in values(data["samples_test"][string(id)])])
        else
            instance.demands_hist[_id] = FloatFromAny(data["demand_hist"][id])
            instance.demands_eval[_id] = Dict{Int, Vector{Float64}}()
            for (scenario, demand_scenario) in enumerate(data["demand_eval"][id])
                instance.demands_eval[_id][scenario] = FloatFromAny(demand_scenario)
            end
            instance.demands_test[_id] = FloatFromAny(data["demand_test"][id])
        end
        
        instance.holding_cost[_id] = data["holding_cost"][id]
        instance.penalty_cost[_id] = data["holding_cost"][id] * penalty_inv
        
        instance.vertices[_id] = Vertex(_id, data["x"][id], data["y"][id])
        
        if pattern == "normal"
            d = Normal(data["mean_demand"][id], data["std_demand"][id])
            instance.sample_demands[_id] = truncated(d, 0., instance.max_inventory[_id])           
        elseif pattern == "uniform"
            d = Uniform(0., 0.5 * instance.max_inventory[_id])
            instance.sample_demands[_id] = d 
        elseif pattern == "bimodal"
            d = MixtureModel([truncated(Normal(data["mean_demand"][id][1], data["std_demand"][id][1]), 0., instance.max_inventory[_id]),
                              truncated(Normal(data["mean_demand"][id][2], data["std_demand"][id][2]), 0., instance.max_inventory[_id])
                    ], [0.5, 0.5])
            instance.sample_demands[_id] = d 
        elseif pattern == "contextual"
            d = collect(values(data["samples_hist"][string(id)]))
            instance.sample_demands[_id] = d 
        end
        
    end

    instance.distances = computeDistances(instance.vertices)
end