#include "Block.h"

// Distance of a point to the closest point in the block
double Blocks::distance(const Point & p1)
{
	double minDistance = 1.e30;
	for (int i2 = 0; i2 < verticesPoints.size(); i2++)
	{
		Point & p2 = verticesPoints[i2];
		Point & p3 = verticesPoints[(i2 + 1) % verticesPoints.size()];
		double dist = Point::distance(p1, p2, p3);
		if (dist < minDistance) minDistance = dist;
	}
	return minDistance;
};

void to_json(json& j, const Blocks& b) 
{
	json jprop = json{
	{"ID", b.id},
	//{"NAME", b.zone_name},
	{"POPULATION", b.nbInhabitants},
	{"AREA", b.area},
	{"PERIMETER", b.perimeter},
	{"DENSITY", b.density},
	{"DIST_DEPOT", b.distDepot},
	{"LIST_ADJACENT", b.adjacentBlockss},
	{"DIST_EUCL", b.distanceEucl},
	{"DIST_GRAPH", b.distanceGraph}
	};
	
	json jgeom = json{
	{"type", "Polygon"},
	{"coordinates", {b.verticesLongLat}}
	};

	j = json{
	{"type", "Feature"},
	{"properties", jprop},
	{"geometry", jgeom}
	};
}
