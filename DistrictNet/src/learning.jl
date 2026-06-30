# =============================================================================
# Model Construction Functions
# =============================================================================

"""
    build_gnn_districtNet_model(graph, hidden_size)

Constructs a GNN model suitable for the districtNet strategy.

# Arguments
- `graph`: The graph for which the model is being built.
- `hidden_size`: The size of the hidden layers.

# Returns
- A Graph Neural Network model configured for districtNet districting problems.
"""

function build_gnn_districtNet_model(graph, hidden_size)
    dim_in = length(props(graph, 1))-1
    dim_out = 1
    AddResidual(l) = Parallel(+, identity, l)
    model = Chain(
        GNNChain(
            GraphConv(dim_in => hidden_size, relu; aggr=mean), 
            GraphConv(hidden_size => hidden_size, relu; aggr=mean),
            GraphConv(hidden_size => hidden_size; aggr=mean),
            Dense(hidden_size => hidden_size, relu),
            Dense(hidden_size => hidden_size),
            Dense(hidden_size => dim_out),
            vec,
        ),    x -> x.ndata.x,
        )
    return model
end

"""
    build_gnn_model(graph, strategy::String, hidden_size::Int)

Builds a GNN model based on the specified strategy (either 'twoStage' or 'districtNet').

# Arguments
- `graph`: The graph for which the model is being built.
- `strategy::String`: The strategy to use ('twoStage' or 'districtNet').
- `hidden_size::Int`: The size of the hidden layers.

# Returns
- A Graph Neural Network model according to the chosen strategy.

# Errors
Throws `ArgumentError` if the strategy is not recognized.
"""

function build_gnn_model(graph, strategy::String, hidden_size::Int)
    if strategy == "districtNet"
        return build_gnn_districtNet_model(graph, hidden_size)
    else
        throw(ArgumentError("Strategy must be either 'Glm' or 'districtNet'."))
    end
end



# =============================================================================
# Training Functions
# =============================================================================
"""
    GNNtrainer_districtNet(model, data_train, data_test, hyperparams)

Trains a GNN model using the districtNet strategy.

# Arguments
- `model`: The GNN model to be trained.
- `data_train`: The training data.
- `data_test`: The testing data.
- `hyperparams`: Hyperparameters for training.

# Returns
- The trained model.
"""
function decayed_epsilon(epoch, initial_epsilon, decay_rate)
    epsilon = initial_epsilon * exp(-decay_rate * epoch)
    if epsilon < 0.001
        epsilon = 0.001
    end
    return initial_epsilon
end
function GNNtrainer_districtNet(model, data_train, data_test, hyperparams)
    start_time = time()
    # Extract hyperparameters
    batch_size, hidden_size, nb_samples, epsilon, weight_decay, initial_lr, nb_epochs, mc_samples= get_hyperparams(hyperparams)
    # Define regularized predictor and loss function
    regularized_predictor = PerturbedAdditive(cmst_exact_solver; ε=epsilon, nb_samples=nb_samples, seed=1234, is_parallel=true)
    
    loss = FenchelYoungLoss(regularized_predictor)

    # Loss function for Flux
    flux_loss(x, y) = loss(model(x.gnn_graph), y; instance=x.instance, subgraphs=x.subgraphs) 

    # Compute total loss for a batch
    flux_loss_batch(batch) = sum(flux_loss(item[1], item[2]) for item in batch) 

    # Tracking losses
    training_losses = Float64[]
    test_losses = Float64[]
    data_train = randomize_data(data_train, mc_samples)
    train_dataset = Flux.DataLoader(data_train, batchsize=batch_size, shuffle=true)
    # DataLoader and optimizer
    opt = Flux.Optimise.ADAM(initial_lr)
    LinearAlgebra.BLAS.set_num_threads(10)
    SPO_CALLS[] = 0; SPO_TIME[] = 0.0   # [TIMING] count CMST set-partition solves during training
    for epoch in 1:nb_epochs
        ep_start = time()                                       # [TIMING] per-epoch (FY + PerturbedAdditive CMST sampling)
        Flux.trainmode!(model, true)
        opt.eta = decayed_learning_rate(epoch, initial_lr)
        # Training
        train_loss = 0
        for batch in train_dataset
            batch_loss = 0
            grads = gradient(Flux.params(model)) do
                batch_loss  = flux_loss_batch(batch)
            end
            train_loss += batch_loss
            Flux.update!(opt, Flux.params(model), grads)
        end

        # Normalize training loss and track
        train_loss /= length(data_train)
        println("Epoch $epoch: train loss = $train_loss  [TIMING] epoch_time = $(round(time()-ep_start, digits=2)) s")
        push!(training_losses, train_loss)
    end
    end_time = time()

    @info "Training done in $(end_time - start_time) seconds"
    _avg = SPO_CALLS[] > 0 ? SPO_TIME[] / SPO_CALLS[] * 1000 : 0.0
    println("[TIMING] DISTRICTNET-FY CMST(set-partition GLPK): calls=$(SPO_CALLS[]) total=$(round(SPO_TIME[],digits=2))s avg=$(round(_avg,digits=2))ms  (perturbed: nb_samples per FY gradient)")
    return model
end





"""
    train_GNN(strategy::String, model, train_data, test_data, hyperparams)

Trains a GNN model based on the specified strategy.

# Arguments
- `strategy::String`: The training strategy ('twoStage' or 'districtNet').
- `model`: The GNN model to train.
- `train_data`: The training data.
- `test_data`: The testing data.
- `hyperparams`: Hyperparameters for training.

# Returns
- The trained GNN model.

# Errors
Throws `ArgumentError` if the strategy is not recognized.
"""

function train_GNN(strategy::String, model, train_data, test_data, hyperparams)
    if strategy == "districtNet"
        return GNNtrainer_districtNet(model, train_data, test_data, hyperparams)
    else
        throw(ArgumentError("Invalid strategy: $strategy"))
    end
end


# =============================================================================
# Utility Functions
# =============================================================================

"""
    decayed_learning_rate(epoch, initial_lr, decay_steps=10, decay_rate=0.9, min_lr=0.0001)

Calculates the decayed learning rate based on the current epoch.

# Arguments
- `epoch`: The current epoch number.
- `initial_lr`: The initial learning rate.
- `decay_steps`: The number of steps after which the rate decays.
- `decay_rate`: The decay rate.
- `min_lr`: The minimum learning rate.

# Returns
- The adjusted learning rate.
"""

function decayed_learning_rate(epoch, initial_lr,
                               decay_steps=10, decay_rate=0.9, min_lr=0.0001)
    lr = max(initial_lr * (decay_rate ^ floor(epoch / decay_steps)), min_lr)
    return lr
end


"""
randomized_constructor(solution)

Constructs a randomized solution using Monte Carlo simulation.

# Arguments
- `solution`: The initial solution.

# Returns
- A randomized CMST solution.
"""
function randomized_constructor(solution, mc_samples)
    # retrun y as the monte carlo solution of randomized constructure
    y = zeros(ne(solution.instance.graph))
    for i in 1:mc_samples
        W = rand(ne(solution.instance.graph))
        update_edge_weights!(solution.instance.graph, W)
        y_rand = get_solution_edges(solution)
        y += y_rand
    end
    # average the solution
    y = y/mc_samples
    return y
end

function randomize_data(data, mc_samples=1)
    temps_data = []
    for item in data
        y = randomized_constructor(item[1].solution, mc_samples)
        push!(temps_data, (item[1], y))
    end
    return temps_data
end



"""
    predict_districtNet_theta(instance, model, mean, std)

Predicts the model outputs for a given instance using a districtNet GNN model.

# Arguments
- `instance`: The problem instance.
- `model`: The trained GNN model.
- `mean`: The mean used for feature normalization.
- `std`: The standard deviation used for feature normalization.

# Returns
- The model's prediction for the given instance.
"""

function predict_districtNet_theta(instance, model, tmean, tstd)
    features = get_instance_features(instance)
    features = (features .- tmean) ./ tstd
    input = create_edge_graph(instance.graph, features)
    theta = model(input)
    theta = reshape(-theta, :,)
    return theta
end


"""
    predict_theta(instance, strategy, model, mean=0, std=1)

Predicts the model outputs for a given instance based on the specified strategy.

# Arguments
- `instance`: The problem instance.
- `strategy`: The prediction strategy ('twoStage' or 'districtNet').
- `model`: The trained GNN model.
- `mean`: The mean used for normalization (default is 0).
- `std`: The standard deviation used for normalization (default is 1).

# Returns
- The model's prediction for the given instance.

# Errors
Throws `ArgumentError` if the strategy is not recognized.
"""

function predict_theta(instance, strategy, model, tmean=0, tstd=1)
    if strategy == "districtNet"
        return predict_districtNet_theta(instance, model, tmean, tstd)
    else
        throw(ArgumentError("Invalid strategy: $strategy"))
    end
end


"""
    get_hyperparams(hyperparams:: Hyperparams)

Extracts the hyperparameters from a `Hyperparams` object.

# Arguments
- `hyperparams`: The hyperparameters object.

# Returns
- A tuple of hyperparameters.
"""

function get_hyperparams(hyperparams:: Hyperparams)
    return hyperparams.batch_size, hyperparams.hidden_size, hyperparams.nb_samples, hyperparams.epsilon, hyperparams.weight_decay, hyperparams.initial_lr, hyperparams.nb_epochs, hyperparams.mc_samples
end
