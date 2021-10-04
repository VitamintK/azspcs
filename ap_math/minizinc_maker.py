import main

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
args = parser.parse_args()

n = args.n

constraints = []
sol = main.APMathSolution(n)
for coord1 in sol.active_coords:
    for coord2 in sol.active_coords:
        if coord1==coord2:
            continue
        r1,c1 = coord1
        r2,c2 = coord2
        if (r1-r2)%2==0 and (c1-c2)%2==0 and ((r1!=r2) or (c1!=c2)):
            constraints.append(f"constraint not ans[{r1},{c1}] \\/ not ans[{r2},{c2}] \\/ not ans[{(r1+r2)//2},{(c1+c2)//2}];")
print('\n'.join(constraints))