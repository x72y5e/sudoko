import numpy as np
from numba import jit
import time
import itertools as it
import pickle


# TODO: multiprocessing


def display_grid(grid: np.ndarray) -> None:
    for i in range(9):
        for j in range(9):
            print(grid[i, j], end="  ")
            if (j + 1) % 3 == 0:
                print(" ", end="")
        if (i + 1) % 3 == 0:
            print("\n")
        else:
            print()


def make_grid() -> np.ndarray:
    grid = np.zeros((9, 9), dtype=np.uint8)
    line = np.arange(1, 10)
    np.random.shuffle(line)
    grid[0] = line
    return grid


def populate(grid: np.ndarray, n: int) -> np.ndarray:
    # n is the number of the line to instantiate
    line = np.arange(1, 10)
    np.random.shuffle(line)
    grid[n] = line
    return grid


@jit
def count_collisions(grid: np.ndarray, n: int) -> int:
    # n is the number of the last populated line
    populated_lines = n + 1
    collisions = 0

    # vertical lines
    for j in range(9):
        collisions += populated_lines - len(np.unique(grid[:populated_lines, j]))

    # boxes
    full_boxes = populated_lines // 3
    part_box = populated_lines % 3

    for i in range(0, full_boxes * 3, 3):
        for j in (0, 3, 6):
            collisions += 9 - len(np.unique(grid[i:i + 3, j:j + 3]))  # 9 unique values in 9*9 box

    if part_box:
        for j in (0, 3, 6):
            # 3 unique values for each line of part-box (i.e. 3 or 6)
            collisions += (part_box * 3) - len(np.unique(grid[full_boxes * 3:full_boxes * 3 + part_box, j:j + 3]))

    return collisions


@jit#(nopython=True)
def shuffle_line(grid: np.ndarray, n: int) -> np.ndarray:
    # n is the number of the line to shuffle
    best = 81
    while True:
        np.random.shuffle(grid[n])
        c = count_collisions(grid, n)
        if c < best:
            best = c
            if best == 0:
                return grid


@jit#(nopython=True)
def cycle_permutations(grid: np.ndarray, n: int) -> np.ndarray:
    # n is the number of the line to shuffle
    best = 81
    for permutation in it.permutations(grid[n], 9):
        grid[n] = permutation
        c = count_collisions(grid, n)
        if c < best:
            best = c
            if best == 0:
                return grid


def fill_last_line(grid: np.ndarray) -> np.ndarray:
    full = set(range(1, 10))
    for i in range(9):
        missing = full.difference(set(grid[:8, i]))
        grid[8, i] = list(missing)[0]
    return grid


def make_puzzle(grid: np.ndarray, clues: int = 21) -> np.ndarray:
    while True:
        mask = np.random.random((9, 9))
        grid[mask > .9] = 0
        n_clues = len(np.where(grid > 0)[0])
        if n_clues <= clues:
            print("\nMade puzzle with {} clues.\n".format(n_clues))
            return grid


def build_new_puzzle(clues: int) -> np.ndarray:
    grid = make_grid()  # grid with first line instantiated
    high = 1  # highest number line to shuffle
    grid = populate(grid, high)
    t0 = time.time()

    while True:
        print("#")
        if high == 8:
            grid = fill_last_line(grid)
        elif high > 6:
            grid = cycle_permutations(grid, high)
        else:
            grid = shuffle_line(grid, high)
        high += 1
        if np.any(grid[8]):
            break
        grid = populate(grid, high)

    puzzle = make_puzzle(grid, clues)
    t1 = round(time.time() - t0, 1)
    print("({} seconds.)\n\n".format(t1))
    return puzzle


if __name__ == '__main__':
    puzzle = build_new_puzzle(25)


    with open("puzzle.pkl", "wb") as f:
        pickle.dump(puzzle, f)

    print("puzzle:\n")
    display_grid(puzzle)
