using Pkg

# Function to read package names from a file and install them
function install_packages_from_file(filename)
    open(filename, "r") do file
        for line in eachline(file)
            pkg_name = strip(line)
            if !isempty(pkg_name)
                println("Installing package: ", pkg_name)
                Pkg.add(pkg_name)
            end
        end
    end
end

file_path = "lib.txt"
install_packages_from_file(file_path)
