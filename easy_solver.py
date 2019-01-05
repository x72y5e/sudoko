import numpy as np


def easy_solve(grid: np.ndarray) -> np.ndarray:
    blanks = len(np.where(grid == 0)[0])

    while True:
        for i, row in enumerate(grid):
            for j, n in enumerate(row):

                if n == 0:
                    full = set(range(1, 10))
                    vertical = full.difference(set(np.unique(grid[:, j])))
                    horizontal = full.difference(set(np.unique(grid[i])))
                    box = full.difference(set(np.unique(grid[3 * (i // 3):3 * (i // 3) + 3,
                                                             3 * (j // 3):3 * (j // 3) + 3])))

                    intersection = vertical & horizontal & box
                    if len(intersection) == 1:
                        solution = list(intersection)[0]
                        grid[i, j] = solution

        if np.all(grid > 0):
            break
        else:
            blanks1 = len(np.where(grid == 0)[0])
            if blanks1 == blanks:
                break
            blanks = blanks1
    return grid