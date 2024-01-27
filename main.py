import numpy as np
import scipy.stats as stats
from constants import Settings
import benchmarks
import errors
import inspect
import csv
from algorithms import Algorithm, DifferentialEvolution, ParticleSwarm, SelfOrganizingMigrationAO, SelfOrganizingMigrationAA
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from typing import Type


def main():
    dimensions = Settings.DIMENSIONS_2
    runs = 30

    all_results = []
    i = 0
    for name, obj in inspect.getmembers(benchmarks, inspect.isclass):
        if i > 4:
            break
        benchmark = obj
        try:
            if name == 'Benchmark':
                continue
            i = i + 1
            print(f"Running {name}")
            results = run_algorithms(benchmark, dimensions, runs)
            all_results.append({
                'benchmark': name,
                'results': results
            })
            # visualize_benchmark(obj(i))
        except NotImplementedError:
            print(f'{name} is not implemented.')
            continue

    if Settings.DEBUG:
        for resultset in all_results:
            print(f"Benchmark: {resultset['benchmark']}")
            if resultset['results'] is None:
                print("There is not enough evidence to conclude statistically significant differences. Sorting is not possible.")
                continue
            for result in resultset['results']:
                print(f"Algorithm {result['algorithm']}: mean {result['result']}, rank {result['rank']}")

    file_path = f"data_{dimensions}_{runs}.csv"
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'Benchmark', 'DE rand/1/bin', 'DE best/1/bin', 'PSO', 'SOMA AtO', 'SOMA AtA'
        ])
        writer.writeheader()

        for resultset in all_results:
            benchmark = resultset['benchmark']
            results = resultset['results']

            if results is None:
                writer.writerow({'Benchmark': benchmark})
            else:
                row = {'Benchmark': benchmark}
                for result in results:
                    row[result['algorithm']] = f"{result['result']} ({result['rank']})"
                writer.writerow(row)


def run_algorithms(benchmark: Type[benchmarks.Benchmark], dimensions: int, runs=30):
    algorithms = {
        "DE rand/1/bin": (DifferentialEvolution, {
            "benchmark": benchmark,
            "dimensions": dimensions,
            "F": 0.8,
            "CR": 0.9,
            "mutation_type": "rand"
        }),
        "DE best/1/bin": (DifferentialEvolution, {
            "benchmark": benchmark,
            "dimensions": dimensions,
            "F": 0.8,
            "CR": 0.9,
            "mutation_type": "best"
        }),
        "PSO": (ParticleSwarm, {
            "benchmark": benchmark,
            "dimensions": dimensions
        }),
        "SOMA AtO": (SelfOrganizingMigrationAO, {
            "benchmark": benchmark,
            "dimensions": dimensions
        }),
        "SOMA AtA": (SelfOrganizingMigrationAA, {
            "benchmark": benchmark,
            "dimensions": dimensions
        }),
    }
    algorithms: (Type[Algorithm], dict)

    results = np.empty((len(algorithms), runs))
    bar = tqdm(range(len(algorithms) * runs))
    for i, (label, algorithm_definition) in enumerate(algorithms.items()):
        results[i, :] = run_benchmark(algorithm_definition=algorithm_definition, runs=runs, bar=bar)
    bar.close()

    average_results = np.mean(results, axis=1)
    sorted_indices = np.argsort(average_results)
    ranks = {}
    for rank, index in enumerate(sorted_indices, 1):
        ranks[index] = rank

    f_statistic, p_value = stats.friedmanchisquare(*results)
    if Settings.DEBUG:
        print("Friedman Test Statistic:", f_statistic)
        print("P-value:", p_value)
    alpha = 0.05
    if p_value < alpha:
        out = []
        for i, algo in enumerate(algorithms.keys()):
            out.append({
                "algorithm": algo,
                "result": average_results[i],
                "rank": ranks[i]
            })
        return out
    else:
        return None


def run_benchmark(algorithm_definition: (Type[Algorithm], dict), runs=30, bar=None):
    best_costs = np.empty(runs, dtype=object)
    mean_costs = np.empty(runs)

    if Settings.MULTIPROCESS:
        with multiprocessing.Pool(processes=Settings.NUM_PROCESSES) as pool:
            results = []
            for i in range(runs):
                result = pool.apply_async(run_worker, (algorithm_definition, i))
                results.append(result)
            for i, result in enumerate(results):
                result.wait()
                bar.update(1)
                mean_costs[i] = np.min(result.get())
    else:
        class_name, args = algorithm_definition
        algorithm = class_name(**args)
        for i in range(runs):
            current_best_costs = []
            try:
                algorithm.restart()
                while True:
                    current_best_costs.append(algorithm.best_individual_cost)
                    algorithm.evolve()
            except StopIteration:
                print(f"Reached max cost evaluations of {algorithm.benchmark.max_runs} runs") if Settings.DEBUG else None
            except errors.StagnantPopulationError:
                print(f"Population is not diverse enough: {algorithm.population}") if Settings.DEBUG else None
            finally:
                bar.update(1) if bar is not None else None
                best_costs[i] = current_best_costs
                mean_costs[i] = np.min(current_best_costs)

                if Settings.DEBUG:
                    print(f"Best cost: {min(algorithm.costs)}")
                    best_index = np.argmin(algorithm.costs)
                    print(f"Best element: {algorithm.population[best_index]}")
    return mean_costs


def visualize_benchmark(benchmark: benchmarks.Benchmark):
    callback = benchmark.evaluate
    step = (benchmark.get_upper_bound() - benchmark.get_lower_bound()) / 100
    x = np.arange(benchmark.get_lower_bound(), benchmark.get_upper_bound(), step)
    y = np.arange(benchmark.get_lower_bound(), benchmark.get_upper_bound(), step)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = callback(element=np.array([x[i], y[j]]))

    cmap = "plasma"

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap=cmap)
    filename = f'images/benchmarks/{benchmark.get_index()}_{benchmark.get_name()}_3d.png'
    plt.savefig(filename)

    fig, ax = plt.subplots()
    levels = np.linspace(Z.min(), Z.max(), 20)
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    filename = f'images/benchmarks/{benchmark.get_index()}_{benchmark.get_name()}_2d.png'
    plt.savefig(filename)
    plt.close()


def run_worker(algorithm_definition: (Type[Algorithm], dict), i):
    class_name, args = algorithm_definition
    algorithm = class_name(**args)
    current_best_costs = []
    try:
        algorithm.restart()
        while True:
            current_best_costs.append(algorithm.best_individual_cost)
            algorithm.evolve()
    except StopIteration:
        print(f"Reached max cost evaluations of {algorithm.benchmark.max_runs} runs") if Settings.DEBUG else None
    except errors.StagnantPopulationError:
        print(f"Population is not diverse enough: {algorithm.population}") if Settings.DEBUG else None
    finally:
        return current_best_costs


if __name__ == "__main__":
    main()
