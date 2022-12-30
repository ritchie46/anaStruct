import time
import subprocess
import os
from copy import deepcopy
from anastruct.fem.examples.ex_8_non_linear_portal import ss

ELEMENT_MAP = deepcopy(ss.element_map)
min_ = 1e8
n = 25
save = True

for i in range(n):
    t0 = time.time()
    ss.solve(verbosity=1)
    ss.element_map = deepcopy(ELEMENT_MAP)
    t = time.time() - t0
    print(t)
    min_ = min(min_, t)

print(f"Best of {n} = {min_} s.")

if save:
    with open("non-linear-solve.csv", "a", encoding="UTF-8") as f:
        os.chdir("../../..")
        git_label = (
            str(subprocess.check_output(["git", "describe", "--tags"]))
            .replace("\\n", "")
            .replace("'", "")[1:]
        )
        f.write(f"{git_label}, {min_}\n")
