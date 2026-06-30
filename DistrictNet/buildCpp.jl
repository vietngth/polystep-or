using CxxWrap
prefix_path = CxxWrap.prefix_path()
project_dir = @__DIR__
deps_dir = joinpath(project_dir, "deps")


function run_build_script(cpp_dir)
    build_dir = joinpath(cpp_dir, "build")
    # Create the build directory if it doesn't exist
    isdir(build_dir) || mkdir(build_dir)

    cd(build_dir) do
        # Run cmake and make commands within the build directory
        run(`cmake -DCMAKE_PREFIX_PATH=$prefix_path $cpp_dir`)
        run(`make`)
    end
end

library_names = ["lkh", "libCostEvaluator.so", "GenerateScenario.so"]
library_paths = [
    joinpath(deps_dir, "LKH"),
    joinpath(deps_dir, "Evaluator"),
    joinpath(deps_dir, "Scenario"),
]

# Only run the build script if the library has not been built yet
for (library_name, library_path) in zip(library_names, library_paths)
    if !isfile(joinpath(library_path, "build", library_name))
        run_build_script(library_path)
    end
end
