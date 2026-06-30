#include "Params.h"
#include <omp.h>
#include <stdio.h>

vector<string> split(const string &s, const char &c)
{
	string buff{""};
	vector<string> v;

	for (auto n : s)
	{
		if (n != c)
			buff += n;
		else if (n == c && buff != "")
		{
			v.push_back(buff);
			buff = "";
		}
	}
	if (buff != "")
		v.push_back(buff);

	return v;
}

Params::Params(string Instance_evalution)
{
	vector<string> DataConfig = split(Instance_evalution, '_');
	dataName = DataConfig[0];//"London"; //string(argv[1]).c_str();
	cout << "CITY NAME: " << dataName << endl;

	targetSizeDistricts = stoi(DataConfig[3]);//3;// stoi(argv[2]);

	minSizeDistricts = floor(0.8 * targetSizeDistricts);
	maxSizeDistricts = ceil(1.2 * targetSizeDistricts);

	cout << "[MIN,TARGET,MAX] SIZE DISTRICTS: "
		 << "[" << minSizeDistricts << "," << targetSizeDistricts << "," << maxSizeDistricts << "]" << endl;


	depotPosition = DataConfig[1];//"C";//string(argv[3]).c_str();
	cout << "DEPOT POSITION: " << depotPosition << endl;

	instanceSize = stoi(DataConfig[2]);//60 ; //stoi(argv[4]);
	cout << "SIZE OF PROBLEM: " << instanceSize << endl;

	seed = 0;//stoi(argv[5]);
	generator.seed(seed);
	cout << "SEED: " << seed << endl;

	nbScenarios = stoi(DataConfig[4]);
	instanceName = dataName;
	cout<<"readBlockssJSON"<<endl;
	readBlockssJSON();
	cout<<"readScenarionJSON"<<endl;
	readScenarionJSON();
	outputName = dataName+"_"+depotPosition+"_"+to_string(instanceSize)+"_"+to_string(targetSizeDistricts) ; 

}


void Params::readScenarionJSON()
{
	string scenarioFileName = dataName + "_" + depotPosition + "_" 
	                        + to_string(instanceSize) + "_" 
							+ to_string(targetSizeDistricts) + ".json";
	ifstream myfile;
	string fileName = "deps/Scenario/output/" + scenarioFileName;
	cout<<"fileName"<<fileName<<endl;
	myfile.open(fileName);

	if (myfile.is_open())
	{	
		json fileContent;
		
		myfile >> fileContent;
		myfile.close();
		
		for (auto &block : fileContent.at("blocks"))
		{
			int blockId = block.at("ID");
			
			blocks[blockId].distDepot = block.at("DEPOT_DIST");
			blocks[blockId].testScenarios = vector<vector<Point>>();

			for (auto &scenario : block.at("Scenarios"))
			{
				vector<Point> scenarioPoints = vector<Point>();
				for (auto &point : scenario)
				{
					Point randomPoint = { point[0],  point[1]};
					scenarioPoints.push_back(randomPoint);
				}

				blocks[blockId].testScenarios.push_back(scenarioPoints);
			}
			
		}			
		
		vector<double> depot = fileContent.at("metadata").at("DEPOT_XY");
		depotPoint = {depot[0], depot[1]};
	}
	else
		throw std::invalid_argument("Impossible to open output file: 2 " + scenarioFileName);
}


vector<District> Params::getDistrictsFromSolution(vector<set<int>> district)
{
	vector<District> districts = vector<District>();
		
	
	for(int i = 0; i < district.size(); i++)
	{
		set<int> blocks  = district[i];
		District d = District(blocks);
		districts.push_back(d);
	}
	return districts;
}




void Params::generateTSPinstance(vector<Point> &points, int scenarioNumber)
{
	ofstream myfile;
	myfile.open(outputName +to_string(scenarioNumber)+ ".tsp");
	if (myfile.is_open())
	{
		myfile << "NAME : TSP" << endl;
		myfile << "COMMENT : Districting Scenario Evaluation" << endl;
		myfile << "TYPE : TSP" << endl;
		myfile << "DIMENSION : " << points.size() << endl;
		myfile << "EDGE_WEIGHT_TYPE: EUC_2D" << endl;
		myfile << "NODE_COORD_SECTION" << endl;
		for (int i = 0; i < points.size(); i++)
			myfile << i + 1 << " " << (int)(1000. * (points[i].x - depotPoint.x)) << " " << (int)(1000. * (points[i].y - depotPoint.y)) << endl;
		myfile.close();
	}
	else
		throw std::invalid_argument("Impossible to create TSP file");
    
	myfile.open(outputName +to_string(scenarioNumber)+".par");
	if (myfile.is_open())
	{
		myfile << "PROBLEM_FILE = " << outputName + to_string(scenarioNumber)<< ".tsp" << endl;
		myfile << "MOVE_TYPE = 5" << endl;
		myfile << "PATCHING_C = 3" << endl;
		myfile << "PATCHING_A = 2" << endl;
		myfile << "RUNS = 1" << endl;
		myfile << "TRACE_LEVEL = 0" << endl;
		myfile << "GAIN23 = NO" << endl;
		myfile << "MAX_TRIALS = 5" << endl;
		myfile << "ASCENT_CANDIDATES = 5" << endl;
		myfile.close();
	}
	else
		throw std::invalid_argument("Impossible to create PAR file");
}

void Params::readBlockssJSON()
{
	string fileName = "data/geojson/" + instanceName + ".geojson";

	ifstream myfile;
	myfile.open(fileName);
	if (myfile.is_open())
	{	
		json fileContent;
		
		myfile >> fileContent;
		myfile.close();

		for (auto& currentBlocks : fileContent.at("features"))
		{
			Blocks block;
			block.id = currentBlocks.at("properties").at("ID");
			if(block.id > instanceSize-1) continue;
			block.verticesPoints = vector<Point>();
			if (currentBlocks.at("geometry").at("type") == "Polygon")
				for (auto& verticePoint : currentBlocks.at("geometry").at("coordinates")[0])
					block.verticesPoints.push_back({verticePoint[0], verticePoint[1]});
			else
				for (auto& verticePoint : currentBlocks.at("geometry").at("coordinates")[0][0])
					block.verticesPoints.push_back({verticePoint[0], verticePoint[1]});
			vector<int> adjacentBlockss = currentBlocks.at("properties").at("LIST_ADJACENT");

			block.adjacentBlockss = set<int>();
			for(int adjBlocks: adjacentBlockss)
			{
				if(adjBlocks > instanceSize-1) continue;
				block.adjacentBlockss.insert(adjBlocks);
			}
			
			this->blocks.push_back(block);
		}
	}

	else
		throw std::invalid_argument("Impossible to open output file: " + fileName);
}
