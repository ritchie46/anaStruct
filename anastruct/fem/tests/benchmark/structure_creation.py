from anastruct.fem.system import SystemElements
import time
import os
import subprocess

save = False
min_ = 1e8
n = 25

for i in range(n):
    t0 = time.time()
    ss = SystemElements()
    el = ss.add_multiple_elements([[0, 0], [10, 10]], n=500)
    ss.point_load(ss.node_map.values(), Fz=1)
    ss.q_load(1, el)
    t = time.time() - t0
    print(t)
    min_ = min(min_, t)

print(f"Best of {n} = {min_} s.")

if save:
    with open("system-creation.csv", "a") as f:
        os.chdir("../../..")
        git_label = str(subprocess.check_output(["git", "describe", "--tags"])).replace('\\n', '').replace("'", "")[1:]
        f.write(f"{git_label}, {min_}\n")