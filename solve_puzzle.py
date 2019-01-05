import numpy as np
from numba import jit
import time
import itertools as it
import pickle
from typing import Set
import sys
from multiprocessing import Process, Queue, cpu_count
from easy_solver import easy_solve
from puzzle_maker import make_puzzle, display_grid


def find_missing_nums(grid: np.ndarray, row: int) -> Set:
    # find the missing numbers from the row
    full = set(range(1, 10))
    present = set(np.unique(grid[row]))
    missing = full.difference(present)
    return missing


#@jit#(nopython=True)
def search(grid: np.ndarray, q: Queue) -> None:

    t0 = time.time()
    # first try to solve with obvious candidates
    grid = easy_solve(grid)
    if np.all(grid > 0):
        print("solved.")
        q.put(grid)

    # otherwise, try to fill the grid, row by row
    working_grid = grid.copy()

    iterations = 0
    done = False
    row = 0

    while not done:
        row_done = False

        if np.all(working_grid[row] > 0):
            row += 1
            if row == 9:
                q.put(working_grid)
            continue
        missing = list(find_missing_nums(working_grid, row)) # this returns a set of the numbers not present in this row
        gaps = np.where(working_grid[row] == 0)[0]  # the positions of the missing numbers
        best = 81
        permutations = list(it.permutations(missing, len(missing)))
        np.random.shuffle(permutations)
        for permutation in permutations:
            iterations += 1
            for pos, num in zip(gaps, permutation):
                working_grid[row, pos] = num
            c = count_collisions(working_grid)
            if c < best:
                best = c
            if c == 0:
                row += 1
                row_done = True
                done = True if row == 9 else False
                break

        if not row_done:
            row = 0
            working_grid = grid.copy()

    t1 = round(time.time() - t0, 1)
    print("\n\nsolution found with {} collisions in {} seconds.\n".format(count_collisions(working_grid), t1))
    q.put(working_grid)


#@jit#(nopython=True)
def count_collisions(grid: np.ndarray) -> int:
    collisions = 0

    # vertical lines
    for j in range(9):
        n = len(grid[:, j][grid[:, j] > 0])  # count numbers already present
        u = len(np.unique(grid[:, j][grid[:, j] > 0]))  # count unique numbers in column
        collisions += n - u

    # boxes
    for i in range(0, 9, 3):
        for j in (0, 3, 6):
            n = len(np.where(grid[3 * (i // 3):3 * (i // 3) + 3,  # count numbers already present
                                  3 * (j // 3):3 * (j // 3) + 3] > 0)[0])  # count the dim1 coords
            u = len(np.unique(grid[3 * (i // 3):3 * (i // 3) + 3,  # count unique numbers in box
                                   3 * (j // 3):3 * (j // 3) + 3][grid[3 * (i // 3):3 * (i // 3) + 3,
                                                                       3 * (j // 3):3 * (j // 3) + 3] > 0]))

            collisions += n - u

    return collisions


def await_solution(puzzle: np.ndarray, q: Queue) -> None:
    solved = q.get()
    print("\nOriginal:\n")
    display_grid(puzzle)
    print("\nSolved:\n")
    display_grid(solved)


if __name__ == '__main__':
    try:
        with open("puzzle.pkl", "rb") as f:
            puzzle = pickle.load(f)
    except FileNotFoundError:
        print("unable to load puzzle.")
        sys.exit(1)

    print("puzzle:\n")
    display_grid(puzzle)
    input("press any key to solve.\n\n")
    q = Queue()
    processes = [Process(target=search, args=(puzzle.copy(), q))
                 for _ in range(7)]
    for p in processes:
        p.start()

    await_solution(puzzle, q)
    for p in processes:
        p.terminate()
        p.join()
    sys.exit(0)
