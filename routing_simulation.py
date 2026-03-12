"""Routing simulation using OR-Tools CVRP solver.

This script simulates a set of bins with stochastic fill-level arrivals and
computes routes to collect bins that cross a fill threshold. It's a simplified
demonstration.
"""
import random
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# generate random coordinates within a bounding box
def generate_bins(n, bbox=(0,0,100,100)):
    x0,y0,x1,y1 = bbox
    bins = []
    for i in range(n):
        x = random.uniform(x0,x1)
        y = random.uniform(y0,y1)
        # capacity liters
        cap = random.choice([50, 80, 100, 120])
        # current fill percentage
        fill = random.uniform(10,70)
        bins.append({'id': i, 'x': x, 'y': y, 'cap': cap, 'fill': fill})
    return bins

def distance(a,b):
    return math.hypot(a['x']-b['x'], a['y']-b['y'])

def create_data_model(locations, demands, vehicle_capacity):
    data = {}
    data['locations'] = locations
    data['num_locations'] = len(locations)
    data['demands'] = demands
    data['num_vehicles'] = 1
    data['vehicle_capacities'] = [vehicle_capacity]
    data['depot'] = 0
    return data

def solve_cvrp(data):
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        a = data['locations'][from_node]
        b = data['locations'][to_node]
        return int(math.hypot(a[0]-b[0], a[1]-b[1]) * 1000)  # scaled
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(data['demands'][from_node])
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route
    else:
        return None

def main():
    random.seed(42)
    bins = generate_bins(25, bbox=(0,0,100,100))
    # depot at (50,50)
    depot = {'id':'depot','x':50,'y':50}
    # Simulate incoming waste: increment fill
    for b in bins:
        b['fill'] += random.uniform(0,50)
    to_collect = [b for b in bins if b['fill'] >= 80.0]
    print(f'{len(to_collect)} bins flagged for collection (threshold 80%)')
    if len(to_collect)==0:
        print('No bins to collect. Exiting.')
        return
    locations = [(depot['x'],depot['y'])] + [(b['x'],b['y']) for b in to_collect]
    # demand in liters = fill% * cap / 100
    demands = [0] + [int(b['fill']/100.0 * b['cap']) for b in to_collect]
    vehicle_capacity = 2000
    data = create_data_model(locations, demands, vehicle_capacity)
    route = solve_cvrp(data)
    if route:
        print('Computed route (indices into locations list):', route)
        print('Route coordinates:')
        for idx in route:
            print(locations[idx])
    else:
        print('No solution found.')

if __name__=='__main__':
    main()
