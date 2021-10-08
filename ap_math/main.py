import sys
sys.path.insert(1, '../')

from collections import defaultdict
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
            self._hash = from_grid._hash
            self._taboos = copy.copy(from_grid._taboos)
            self._taboo_free = copy.copy(from_grid._taboo_free)
        else:
            grid = np.zeros((2*n-1, 2*n-1), np.int8) # or bitset
            self.active_coords = []
            self.live_coords = set()
            self.sc = 0
            self._hash = 0
            self._taboos = defaultdict(int)
            self._taboo_free = set()
            k = 1-n
            for i in range(2*n-1):
                for j in range(min(k,0), max(k,0)):
                    grid[i][j] = -1
                k+=1
            for i in range(2*n-1):
                for j in range(2*n-1):
                    if grid[i][j] != -1:
                        self.active_coords.append((i,j))
                        self._taboos[(i,j)] = 0
                        self._taboo_free.add((i,j))
                        # self._hash |= (1 << self._coord_to_int((i,j)))
            # for i in range(2*n-1):
            #     for j in range(2*n-1):         
            #         if grid[i][j] != -1 and (i,j) not in self.active_coords:
            #             self.active_coords.append((i,j))
            #         if grid[j][i] != -1 and (j,i) not in self.active_coords:
            #             self.active_coords.append((j,i))
            # random.shuffle(self.active_coords)
            # print(self.active_coords)
            # print([r*(n*2-1)+c for r,c in self.active_coords])
            self.active_coord_set = set(self.active_coords)
    def _coord_to_int(self, coord):
        r, c = coord
        return r*(2*n - 1) + c
    def _can_add(self, coord):
        return self._taboos[coord] == 0
        # if coord not in self.active_coord_set:
        #     return False
        # if coord in self.live_coords:
        #     return False
        # for xcoord in self.live_coords:
        #     # d = coord - xcoord
        #     newcoord = (coord[0]*2-xcoord[0], coord[1]*2-xcoord[1])
        #     if newcoord in self.active_coord_set and newcoord in self.live_coords:
        #         return False
        #     if (coord[0]-xcoord[0])%2 == 0 and (coord[1]-xcoord[1])%2==0:
        #         newcoord = ((coord[0]+xcoord[0])//2, (coord[1]+xcoord[1])//2)
        #         if newcoord in self.active_coord_set and newcoord in self.live_coords:
        #             return False
        # return True 
    def _add_taboo(self, coord):
        if coord not in self.active_coord_set:
            return
        self._taboos[coord] += 1
        if coord in self._taboo_free:
            # self._hash ^= (1 << self._coord_to_int(coord))
            self._taboo_free.remove(coord)
    def _remove_taboo(self, coord):
        if coord not in self.active_coord_set:
            return
        self._taboos[coord] -= 1
        if self._taboos[coord] == 0:
            self._taboo_free.add(coord)
            # self._hash |= (1 << self._coord_to_int(coord))
    def add(self, coord):
        self._add_taboo(coord)
        for xcoord in self.live_coords:
            d = (coord[0]-xcoord[0], coord[1]-xcoord[1])
            newcoord = (coord[0] + d[0], coord[1] + d[1])
            self._add_taboo(newcoord)
            d = (-d[0], -d[1])
            newcoord = (xcoord[0] + d[0], xcoord[1] + d[1])
            self._add_taboo(newcoord)
            if (coord[0]-xcoord[0])%2 == 0 and (coord[1]-xcoord[1])%2==0:
                newcoord = ((coord[0]+xcoord[0])//2, (coord[1]+xcoord[1])//2)
                self._add_taboo(newcoord)
        self.sc += 1
        self.live_coords.add(coord)
        # self._hash |= (1 << self._coord_to_int(coord))
    def remove(self, coord):
        self.sc -= 1
        self.live_coords.remove(coord)
        # self._hash ^= (1 << self._coord_to_int(coord))
        self._remove_taboo(coord)
        for xcoord in self.live_coords:
            d = (coord[0]-xcoord[0], coord[1]-xcoord[1])
            newcoord = (coord[0] + d[0], coord[1] + d[1])
            self._remove_taboo(newcoord)
            d = (-d[0], -d[1])
            newcoord = (xcoord[0] + d[0], xcoord[1] + d[1])
            self._remove_taboo(newcoord)
            if (coord[0]-xcoord[0])%2 == 0 and (coord[1]-xcoord[1])%2==0:
                newcoord = ((coord[0]+xcoord[0])//2, (coord[1]+xcoord[1])//2)
                self._remove_taboo(newcoord)
    def get_all_actions(self):
        # ans = []
        return self._taboo_free
        return copy.copy(self._taboo_free)
        # return ans
            
    def sample_neighbor(self, temperature):
        ans = APMathSolution(self.n, self)
        for i in range(3):
            if random.random() < 0.2 * max(1-temperature,0.2) and ans.sc > 0:
                sampled_deletion = random.choice(list(ans.live_coords))
                ans.remove(sampled_deletion)
        if ans.sc == 0 or random.random() < 1:
            sampled_coord = random.choice(self.active_coords)
            if ans._can_add(sampled_coord):
                ans.add(sampled_coord)
#             coords = [(0, 0), (0, 5), (0, 1), (0, 2), (0, 3), (0, 4),
#             (1, 0),  (9, 4), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1), (7, 2), (8, 3),
#             (10, 5), (10, 10),  (10, 6), (10, 7), (10, 8), (10, 9),
#             (1, 6),  (9, 10), (2, 7), (3, 8), (4, 9), (5, 10), (6, 10), (7, 10), (8, 10),
#              (1, 1), (1, 5),  (1, 2), (1, 3), (1, 4), 
#                 (2, 1),  (8, 4), (3, 1), (4, 1), (5, 1), (6, 2), (7, 3),
#              (9, 5),  (9, 9), (9, 6), (9, 7), (9, 8),
#              (2, 6), (8, 9), (3, 7), (4, 8), (5, 9), (6, 9), (7, 9), 

#               (2, 2), (2, 3), (2, 4), (2, 5), 
# (3, 6),
# (4, 7),
# (5, 8),
# (6, 8),
# (7, 8),
#               (8, 5), (8, 6), (8, 7), (8, 8),  
#                 (3, 2),
# (4, 2),
# (5, 2),
# (6, 3),
# (7, 4),
#                (3, 3), (3, 4), (3, 5),   
#                (4, 3), (4, 4), (4, 5), (4, 6),   
#                (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),   
#                (6, 4), (6, 5), (6, 6), (6, 7),   
#                (7, 5), (7, 6), (7, 7)
#             ]
#             for sampled_coord in coords:
#                 if ans._can_add(sampled_coord):
#                     ans.add(sampled_coord)
#                     return ans
        else:
            # sampled_coord = random.choice(self.active_coords)
            # if ans._can_add(sampled_coord):
            #     ans.add(sampled_coord)
            # return ans
            # x,y = random.choice(list(ans.live_coords))
            x,y = random.choice(self.active_coords)
            # pattern = random.choice(([(0,1), (1,1)], [(1,1),(1,0)]))
            pattern = [(0,0), (0,1), (0,3), (0,4), (0,9), (0,10), (0,12), (0,13),
            (1,0), (1,1), (1,3), (1,4), (1,9), (1,10), (1,12), (1,13),]
            # (3,0), (3,1), (3,3), (3,4), (3,9), (3,10), (3,12), (3,13)]
            pattern = [(0,0),(0,1),(1,0),(1,2),(2,1),(2,2)]
            # pattern = random.choice([
            #     [(0,0), (0,1)],
            #     [(0,0), (0,1), (0,3), (0,4)],
            #     [(0,0),(1,0)]
            # ])
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
        return self.score() + len(self._taboo_free)/len(self.active_coord_set)
    def score(self):
        return self.sc
    def apply_action(self, action):
        self.add(action)
    def undo_action(self, action):
        self.remove(action)
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
                elif (r,c) not in self._taboo_free:
                    to_print.append('\033[1;33m0\033[0;39m')
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--n', type=int)
    parser.add_argument('--its', type=int)
    parser.add_argument('--p', type=int)
    parser.add_argument('--k', type=int)
    commandline_args = parser.parse_args()
    t = commandline_args.type
    n = commandline_args.n
    its = commandline_args.its
    p = commandline_args.p
    k = commandline_args.k
    if its is None:
        its = 1000
    if p is None:
        p = 20
    if k is None:
        k = 1000

    initial_solution = APMathSolution(n)
    try:
        with open(f'{directory}/{n}_score.out', 'r') as f:
            best = int(f.readline())
    except FileNotFoundError:
        with open(f'{directory}/{n}_score.out', 'w') as f:
            f.write('0')
        best = 0
    
    if t == 'beam':
        s = common.MonteCarloBeamSearcher(initial_solution, best)
        s.go(its, p, k)
    elif t=='backtrack':
        # s = common.ExhaustiveBacktracker(initial_solution, best)
        # s.go()
        s = common.SamplingBacktracker(initial_solution, best, k=k)
        s.go()
    else:
        raise ValueError(f'{t} is not a recognized type')