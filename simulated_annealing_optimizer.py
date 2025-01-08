from typing import List, Dict, Tuple
import mlrose

class HillClimbOptimizer:
    def __init__(self, people: List[Tuple[str, str]], destination: str, flights: Dict[Tuple[str, str], List[Tuple[str, str, int]]]):
        self.people = people
        self.destination = destination
        self.flights = flights

    def print_flights(self, schedule: List[int]) -> int:
        total_cost = 0
        flight_id = 0

        for i in range(len(self.people)):
            origin = self.people[i][1]

            departure = self.flights[(origin, self.destination)][schedule[flight_id]]
            total_cost += departure[2]
            flight_id += 1

            # Return flight
            return_flight = self.flights[(self.destination, origin)][schedule[flight_id]]
            total_cost += return_flight[2]
            flight_id += 1

        return total_cost

    def optimize(self, max_val: int = 10, init_temp: int = 10000, random_state: int = 1) -> Tuple[List[int], int]:
        fitness = mlrose.CustomFitness(self.print_flights)
        problem = mlrose.DiscreteOpt(length=len(self.people) * 2, fitness_fn=fitness, maximize=False, max_val=max_val)

        best_solution, best_cost = mlrose.simulated_annealing(
            problem,
            schedule=mlrose.decay.GeomDecay(init_temp=init_temp),
            random_state=random_state
        )

        return best_solution, best_cost

def load_flights(file_path: str) -> Dict[Tuple[str, str], List[Tuple[str, str, int]]]:

    flights: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = {}
    with open(file_path, "r") as file:
        for line in file:
            origin, destination, departure, arrival, price = line.strip().split(",")
            flights.setdefault((origin, destination), []).append((departure, arrival, int(price)))
    return flights

if __name__ == "__main__":

    people = [
        ("Lisbon", "LIS"),
        ("Madrid", "MAD"),
        ("Paris", "CDG"),
        ("Dublin", "DUB"),
        ("Brussels", "BRU"),
        ("London", "LHR")
    ]
    destination = "FCO"

    file_path = "../hill-climb/flights.txt"
    flights = load_flights(file_path)

    optimizer = HillClimbOptimizer(people, destination, flights)
    best_solution, best_cost = optimizer.optimize()

    print("Best solution:", best_solution)
    print("Total cost of the best solution:", best_cost)
