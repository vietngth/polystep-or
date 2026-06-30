# DISTRICTNET: Decision-Aware Learning for Geographical Districting


DistrictNet is a structured learning approach that integrates a combinatorial optimization layer into a graph neural network to efficiently solve real-world districting problems.

**Link:** [https://arxiv.org/abs/2412.08287](https://arxiv.org/abs/2412.08287) *(Accepted at NeurIPS 2024)*

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

### Requirements

To run the code, you must have Python 3.6 or higher and Julia 1.7 or higher installed on your machine.



#### Julia Dependencies

First, set up the Julia environment and install the necessary packages. Start by opening Julia from your terminal or command prompt. Then, follow these steps:

1. Activate the Julia environment:
    ```julia
    julia> ]
    ```

    This will switch to the package manager prompt, indicated by `pkg>`.

2. In the package manager prompt, activate the project environment:
    ```julia
    (@v1.10) pkg> activate .
    ```

    This command activates the project's environment, ensuring that the Julia packages will be installed in a local environment specific to this project. 

3. Instantiate the environment to install the dependencies:
    ```julia
    (DistrictNet) pkg> instantiate
    ```

    The `instantiate` command will install all the dependencies listed in the project's `Project.toml` and `Manifest.toml` files.

   Or you can run the following command in the terminal to install the required Julia packages:

```bash
julia install_packages.jl
```
#### Python Dependencies

Next, install the required Python packages listed in `requirements.txt`. Open a terminal or command prompt and run the following command:

```bash
pip install -r requirements.txt
```
#### C++ Dependencies
Ensure a C++ compiler is installed on your system. To compile the required C++ libraries:

1. Execute the following commands:
   ```bash
   julia buildCpp.jl
   ```
## Usage

### Generating Training Data

1. **City Generation**:
   Generate a set of synthetic cities to be used as training data.
   ```bash
   julia generate_cities.jl <n> <data_path> <save_path>
   ```
   - `n`: Number of cities to generate.
   - `data_path`: Path to the original city's GeoJSON data.
   - `save_path`: Directory where the generated city data will be saved.

### Computing District Enumeration and Costs

2. **Cost Computation**:
   Enumerate districts for each city and calculate their costs.
   ```bash
   julia ComputeTSPcost.jl <cityname> 
   ```
   - `cityname`: Name of the city.

### Running Experiments

3. **Types of Experiments**:
   Our framework includes five distinct experiment setups:
   - `City Generalization` (id = 2): Involves training the model using multiple small cities with a fixed district size of 3. The model is then tested with different district sizes in larger cities.
   - `Larger-Size City Generalization` (id = 3): Trains the model with multiple small cities, maintaining a fixed district size of 3, and tests it on cities of the same and larger sizes, to assess scalability.

   - `Multi-Size Training` (id = 4): Involves training the model with multiple cities (fixe the number of data) at a fixed district size of 3, and then testing it on the same and larger city sizes with varying district sizes.
   
   - `Small-City Specialization` (id = 5): This setup trains the model with multiple small cities, using a fixed district size of 3, and tests it with the same and smaller district sizes, focusing on small city.


4. **Experimentation**:
   Conduct experiments on the training dataset.
  
```bash
julia Experiments.jl <experiment_type> <city> <model_module> <action> <district_size> <number_bu> <depot> <number_of_data>
```

#### Arguments:
- `experiment_type`: Type of experiment.
- `city`: Name of the city ('dummy' for a general experiment).
- `model_module`: Name of the model module to use (e.g., `districtNet`, `predictGnn`, `BD`, `FIG`, etc.).
- `action`: Action to perform (e.g., 'train').
- `district_size`: Size of the district.
- `number_bu`: Number of basic units for training.
- `depot`: Location of the depot in the city (default is 'C').
- `number_of_data`: Number of data points used for training (default is 100).

#### Example Command:
```bash
julia Experiments.jl 2 dummy_name districtNet train 3 30 C 100
```


### Solving the Districting Problem

5. **Problem Solving**:
   Apply the model to solve the districting problem.
   
```bash
julia experiments.jl <experiment_type> <city> <model_module> <action> <district_size> <number_bu> <depot> <number_of_data>
```

#### Arguments:
- `experiment_type`: Type of the experiment.
- `city`: Name of the city where the districting problem is being solved.
- `model_module`: Name of the model module to use (e.g., `districtNet`, `predictGnn`, `BD`, `FIG`, etc.).
- `action`: Action to be taken
  - `'solve'`: Solves the districting problem for the given city.
  - `'compute_gap'`: Computes the gap, but only for small instances.
- `district_size`: Desired district size for the solution.
- `number_bu`: Number of basic units (blocks) for the specific city.
- `depot`: Location of the depot (default is 'C').
- `number_of_data`: Amount of data used in training or evaluation (default is 100).

#### Example Command (Solving):
```bash
julia experiments.jl 2 London districtNet solve 6 120 C 100
```

#### Example Command (Gap Computation for Small Instances):
```bash
julia experiments.jl 5 dummy_name districtNet compute_gap 3 30 C 100
```


### Evaluating the Districting Solution

6. **Solution Evaluation**:
You can evaluate the districting solutions using three different evaluation types: `runCity`, `runExp`, or `runAll`.

- `runCity`: Evaluates the solution for a specific city, district size, and experiment type.
- `runAll`: Evaluates the solution for all cities, district sizes, and experiment types.
- `runExp`: Evaluates the solution for a specific experiment type across multiple cities and district sizes.

### Usage

```bash
julia experiment_evaluator.jl <evaluation_type> <experiment_type> <city> <district_size> <number_bu> <depot> <number_of_data>
```
- `evaluation_type`: Type of evaluation to run (`runCity`, `runExp`, or `runAll`).
- `experiment_type`: The type of experiment to evaluate.
- `city`: Name of the city (required for `runCity`).
- `district_size`: Size of the districts (required for `runCity`).
- `number_bu`: Number of basic units for the specified city (required for `runCity`).
- `depot`: Depot location.
- `number_of_data`: Amount of data used in training (optional, default is 100).

### Example Commands

1. **Evaluate a specific city:**

```bash
julia experiment_evaluator.jl runCity 2 London 6 120 C 100
```

2. **Evaluate all cities, district sizes, and experiment types:**

```bash
julia experiment_evaluator.jl runAll
```

3. **Evaluate a specific experiment across multiple cities:**

```bash
julia experiment_evaluator.jl runExp 2 C
```


## License

DistrictNet is licensed under the MIT License. See [LICENSE](https://choosealicense.com/licenses/mit/) for more details.
