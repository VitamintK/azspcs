import sys
sys.path.insert(1, '../')

import math
import random
import numpy as np
import copy
import argparse

from lib import common

directory = 'results'

def setstr(s):
    if len(s) == 0:
        return '{}'
    return str(s)

class APMathSolution(common.Solution):
    def __init__(self, n, from_grid=None):
        self.n = n
        if from_grid is not None:
            # self.grid = from_grid.grid.copy()
            self.active_coords = from_grid.active_coords
            self.active_coord_set = from_grid.active_coord_set
            self.sc = from_grid.sc
            self.live_coords = copy.copy(from_grid.live_coords)
        else:
            grid = np.zeros((2*n-1, 2*n-1), np.int8) # or bitset
            self.active_coords = []
            self.live_coords = set()
            self.sc = 0
            k = 1-n
            for i in range(2*n-1):
                for j in range(min(k,0), max(k,0)):
                    grid[i][j] = -1
                k+=1
            for i in range(2*n-1):
                for j in range(2*n-1):
                    if grid[i][j] != -1:
                        self.active_coords.append((i,j))
                        # self.free_coords.add((i,j))
            random.shuffle(self.active_coords)
            self.active_coord_set = set(self.active_coords)
    def _can_add(self, coord):
        if coord not in self.active_coord_set:
            return False
        if coord in self.live_coords:
            return False
        for xcoord in self.live_coords:
            # d = coord - xcoord
            newcoord = (coord[0]*2-xcoord[0], coord[1]*2-xcoord[1])
            if newcoord in self.active_coord_set and newcoord in self.live_coords:
                return False
            if (coord[0]-xcoord[0])%2 == 0 and (coord[1]-xcoord[1])%2==0:
                newcoord = ((coord[0]+xcoord[0])//2, (coord[1]+xcoord[1])//2)
                if newcoord in self.active_coord_set and newcoord in self.live_coords:
                    return False
        return True 
    def add(self, coord):
        self.sc += 1
        self.live_coords.add(coord)
    def remove(self, coord):
        self.sc -= 1
        self.live_coords.remove(coord)
    def sample_neighbor(self, temperature):
        ans = APMathSolution(self.n, self)
        for i in range(3):
            if random.random() < 0.2 * max(temperature,0.2) and ans.sc > 0:
                sampled_deletion = random.choice(list(ans.live_coords))
                ans.remove(sampled_deletion)
        if ans.sc == 0 and random.random() < 0.3:
            sampled_coord = random.choice(self.active_coords)
            if ans._can_add(sampled_coord):
                ans.add(sampled_coord)
        else:
            # x,y = random.choice(list(ans.live_coords))
            x,y = random.choice(self.active_coords)
            # pattern = random.choice(([(0,1), (1,1)], [(1,1),(1,0)]))
            pattern = [(0,0), (0,1), (1,1)]
            # for x_offset, y_offset in pattern:
            #     # x_offset, y_offset = random.normalvariate(0, 1), random.normalvariate(0, 1)
            #     sampled_coord = (round(x+x_offset), round(y+y_offset))
            #     if not ans._can_add(sampled_coord):
            #         return ans
            for x_offset, y_offset in pattern:
                # x_offset, y_offset = random.normalvariate(0, 1), random.normalvariate(0, 1)
                sampled_coord = (round(x+x_offset), round(y+y_offset))
                if ans._can_add(sampled_coord):
                    ans.add(sampled_coord)
        return ans
    def heuristic(self):
        return self.score() + random.randint(-2,2)
    def score(self):
        return self.sc
    def apply_action(self, action):
        pass
    def serialize(self):
        ans = []
        x = 1-self.n
        for r in range(self.n*2-1):
            row = set()
            for c in range(max(0,x), self.n*2-1 + min(0,x)):
                if (r,c) in self.live_coords:
                    row.add(c-max(0,x))
            ans.append(row)
            x += 1
        return ','.join(setstr(s) for s in ans)
    def pretty(self):
        def color(x):
            if x == '1':
                return '\033[1;35m'+x+'\033[0;39m'
        if self.n > 49:
            return ''
        n = self.n
        max_len = 2*n-1
        k = 1-n
        for r in range(max_len):
            s = ""
            to_print = []
            spaces = abs(max_len//2-r)
            for c in range(max(0,k), self.n*2-1 + min(0,k)):
                if (r,c) in self.live_coords:
                    to_print.append(color('1'))
                else:
                    to_print.append('0')
            # to_print = self.grid[r][max(0,k):min(k,0)+2*n-1]
            print(' '*spaces + ' '.join(str(x) for x in to_print))
            k+=1
    def save(self):
        with open(f'{directory}/{self.n}_repr.out', 'w') as f:
            f.write(self.serialize())
        with open(f'{directory}/{self.n}_score.out', 'w') as f:
            f.write(f'{self.score()}')
        
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--its', type=int)
commandline_args = parser.parse_args()
n = commandline_args.n
its = commandline_args.its
if its is None:
    its = 1000

initial_solution = APMathSolution(n)
try:
    with open(f'{directory}/{n}_score.out', 'r') as f:
        best = int(f.readline())
except FileNotFoundError:
    with open(f'{directory}/{n}_score.out', 'w') as f:
        f.write('0')
    best = 0
s = common.MonteCarloBeamSearcher(initial_solution, best)
s.go(its, 10, 12)