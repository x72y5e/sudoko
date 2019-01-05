import argparse
import sys
import numpy as np
import pickle
from multiprocessing import Process, Queue, cpu_count
from puzzle_maker import build_new_puzzle, display_grid
from solve_puzzle import search, await_solution
from typing import Dict


def _solve(puzzle: np.ndarray, p: int) -> None:
    q = Queue()
    processes = [Process(target=search, args=(puzzle.copy(), q))
                 for _ in range(p)]
    for p in processes:
        p.start()

    await_solution(puzzle, q)
    for p in processes:
        p.terminate()
        p.join()
    sys.exit(0)


def run(args: Dict):

    # generate new puzzle, save, and exit
    if args["new"]:
        puzzle = build_new_puzzle(args["clues"])
        with open("puzzle.pkl", "wb") as f:
            pickle.dump(puzzle, f)
        print("puzzle:\n")
        display_grid(puzzle)
        print("saving...")
        print("done.")
        sys.exit(0)

    # generate new puzzle, save, and solve
    if args["gen"]:
        puzzle = build_new_puzzle(args["clues"])
        with open("puzzle.pkl", "wb") as f:
            pickle.dump(puzzle, f)
        print("puzzle:\n")
        display_grid(puzzle)
        print("saving...")

    # default: load saved puzzle, and solve
    else:
        try:
            with open("puzzle.pkl", "rb") as f:
                puzzle = pickle.load(f)
                print("puzzle:\n")
                display_grid(puzzle)
        except FileNotFoundError:
            print("No puzzle found. Run again with --gen flag to generate new puzzle.")
            sys.exit(1)

    # solve
    cpus = min(cpu_count(), args["parallel"])
    print("solving ({} cores)...".format(cpus))
    _solve(puzzle, cpus)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sudoku")
    parser.add_argument("--new", "-n",
                        const=1,
                        nargs="?",
                        type=int,
                        help="-n to generate new puzzle and save it. Use '--clues' argument to specify number of clues.",
                        default=0)
    parser.add_argument("--gen", "-g",
                        const=1,
                        nargs="?",
                        type=int,
                        help="1 to generate new puzzle (default 0: use saved puzzle, if available).",
                        default=0)
    parser.add_argument("--clues", "-c",
                        type=int,
                        help="Number of clues in new puzzle (default 25).",
                        default=25)
    parser.add_argument("--parallel", "-p",
                        type=int,
                        help="Number of cores to use (default 1).",
                        default=1)

    args = {k: v for k, v in parser.parse_args().__dict__.items()}
    run(args)
