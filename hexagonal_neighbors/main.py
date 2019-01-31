import sys
import argparse

sys.setrecursionlimit(10000)


parser = argparse.ArgumentParser(description='Optimize.')
parser.add_argument('N', metavar='N', type=int, help='n (from 3 to 27 inclusive)')
parser.add_argument('-l', dest='limit', metavar='--limit', type=int)
parser.add_argument('-b', dest='branches', metavar='--branches', help='branching factor', type=int)
args = parser.parse_args()
n = args.N
limit = args.limit
branches = args.branches if args.branches is not None else (n*n)//12+5

import numpy as np

"""Alright, I'm gonna make up some terms right now right here so listen up.
I'm gonna say that "feasible" means a hexgrid that is submittable to the contest.
I'm also gonna say that "potentially feasible" means a hexgrid for which, while not feasible, there exists a set of hexes which we can INCREASE in order to make the hexgrid feasible
"""

import os
import pickle
def get_best(n):
    if not os.path.isfile('./best_{}.txt'.format(n)):
        return None
    else:
        with open('./best_{}.txt'.format(n), 'r') as f:
            #a, b = pickle.load(f)
            a, b = f.readlines()
        return int(a),b



class HexagonGrid:
    def __init__(self, from_hexgrid=None):
        if from_hexgrid is None:
            self.grid = np.ones((2*n-1, 2*n-1), np.int8)
            k = 1-n
            for i in range(2*n-1):
                for j in range(min(k,0), max(k,0)):
                    self.grid[i][j] = -1
                k+=1
            self.active_coords = []
            for i in range(2*n-1):
                for j in range(2*n-1):
                    if self.grid[i][j] != -1:
                        self.active_coords.append((i,j))
            self.sum = len(self.active_coords)
            self.grid_neighbor_count = np.zeros((2*n-1, 2*n-1, 8), np.int8)
            self.build_grid_neighbor_count()
        else:
            self.grid = np.copy(from_hexgrid.grid)
            self.active_coords = from_hexgrid.active_coords[:]
            self.sum = from_hexgrid.sum
            self.grid_neighbor_count = np.copy(from_hexgrid.grid_neighbor_count)

            self.cluster_heuristic_score = from_hexgrid.cluster_heuristic_score
        

        ### diagnostics to see if doing caching or memoization would help with the monte-carlo backtracking
        #self._hashes_seen = set()
        #self._hash_collisions = 0
        #result: on n = 15, was getting about 10+/-10 collisions every 10,000 iterations
        ### end diagnostics   
    def _hash(self):
        """ this is for my testing purpose only.  I think.  It's O(n)"""
        return hash(str(self.grid))

    def build_grid_neighbor_count(self):
        """Assumes that self.grid_neighbor_count is all zeros."""
        for r, c in self.active_coords:
                for nr, nc in self.get_neighbors(r, c):
                    self.grid_neighbor_count[nr][nc][self.grid[r][c]]+=1
    def check_mutate_hex_feasibility(self, row, col, new_value):
        for i in range(1,new_value):
            if self.grid_neighbor_count[row][col][i] == 0:
                return False
        prev_value = self.grid[row][col]
        for nr, nc in self.get_neighbors(row, col):
            if self.grid_neighbor_count[nr][nc][prev_value] == 1 and prev_value < self.grid[nr][nc]:
                return False
        return True
    def mutate_hex(self, row, col, new_value):
        """ ~~I'm calling each little hex a 'pix' now and there's nothing you can do about it ~~ alright fuck it I'm calling them hexes.
        if check_feasibility is True, then mutate_hex will assume that the hexgrid is already feasible and will return False if this mutation would cause unfeasibility"""
        prev_value = self.grid[row][col]
        for nr, nc in self.get_neighbors(row, col):
            self.grid_neighbor_count[nr][nc][prev_value]-=1
            self.grid_neighbor_count[nr][nc][new_value]+=1
        self.grid[row][col] = new_value
        self.sum += new_value - prev_value
        return True
    def increment_hex(self, row, col, check_feasibility=False):
        prev_value = self.grid[row][col]
        if check_feasibility:
            if not self.check_mutate_hex_feasibility(row, col, prev_value+1):
                return False
        self.mutate_hex(row, col, prev_value+1)
        return prev_value #this is icky, should probably be return True, but it's my code and I can do whatever I want.  Man I love python.  It is truly the highest level language.  But with great power comes great responsibility, as my dad always said.  Luckily for someone with as big of a brain as me, great responsibility is no problem.
    def decrement_hex(self, row, col, check_feasibility=False):
        prev_value = self.grid[row][col]
        if check_feasibility:
            if not self.check_mutate_hex_feasibility(row, col, prev_value-1):
                return False
        self.mutate_hex(row, col, prev_value-1)
        return prev_value
    def get_neighbors(self, row, col):
        return ((row+dr, col+dc) for dr,dc in [(-1,-1), (-1,0), (0,-1), (0,1), (1,0), (1,1)] if 0<=row+dr<2*n-1 and 0<=col+dc<2*n-1 and self.grid[row+dr][col+dc]!=-1)

    def monte_carlo_backtrack_start(self, rolls):
        print("~~~\nBeginning monte carlo backtrack with N={} and branching factor {}\n~~~".format(n, rolls))
        self.best_sum_of_all_time = self.sum
        self.iterations = 0
        self._monte_carlo_backtrack(rolls)

    def _monte_carlo_backtrack(self, rolls):
        """ run a simple backtracking algorithm.  Only considers feasible hexgrids.
        Don't feel like doing backtracking sequentially, so let's just consider random hexes to increment."""
        self.iterations+=1
        if self.iterations%10000==0:
            print(self.iterations)
            #print("Hash collisions: {}".format(self._hash_collisions))
        if limit and self.iterations > limit:
            return

        is_dead_end = True
        import random
        for i in range(rolls):
            row, col = random.choice(self.active_coords)
            if not self.increment_hex(row, col, True):
                continue
            else:
                is_dead_end = False
            if self.sum >= self.best_sum_of_all_time - 3:
                ###EXPERIMENT
                #when we're close to something good, explore much more thoroughly
                self._monte_carlo_backtrack(branches*3)
            else:
                self._monte_carlo_backtrack(rolls)
            self.decrement_hex(row, col)

        if is_dead_end:
            #hsh = self._hash()
            #if hsh in self._hashes_seen:
            #    self._hash_collisions+=1
            #self._hashes_seen.add(hsh)

            if self.sum > self.best_sum_of_all_time:
                self.best_sum_of_all_time = self.sum
                print(self.get_raw_score())
                self.pretty_print()
                print(self.serialize())
                best = get_best(n)

                if best is None or self.get_raw_score() > best[0]:
                    print("new all-time-best")
                    #if best is not None:
                    #    print(best)
                    #    print(type(best), type(best[0]), type(best[1]))
                    self.save()
                    print("saved!")
                elif best is not None:
                    print("all-time-best is {} :(".format(best[0]))
    def save(self):
        with open('best_{}.txt'.format(n), 'w') as f:
            #pickle.dump((int(self.get_raw_score()), self.serialize()), f)
            f.writelines([str(self.get_raw_score())+'\n', self.serialize()])
    def serialize(self):
        """this is the format that is expected for submission"""
        return ','.join('('+','.join(str(x) for x in row if x!=-1)+')' for row in self.grid)
    def get_raw_score(self):
        return self.sum - 3*n*n + 3*n - 1
    def pretty_print(self):
        def color(x):
            if x == '7':
                return '\033[1;33m'+x+'\033[0;39m'
            elif x == '6':
                return '\033[1;32m'+x+'\033[0;39m'
            elif x == '5':
                return '\033[1;31m'+x+'\033[0;39m'
            elif x == '4':
                return '\033[1;35m'+x+'\033[0;39m'
            else:
                return x
        max_len = 2*n-1
        offset = False
        k = 1-n
        for r in range(max_len):
            s = ""
            spaces = abs(max_len//2-r)
            to_print = self.grid[r][max(0,k):min(k,0)+2*n-1]
            print(' '*spaces + ' '.join(color(str(x)) for x in to_print))
            k+=1

import random
class RaySearchFeasible:
    def __init__(self):
        self.lorde = []
        self.best_sum_of_all_time = 0

    def ray_search_feasible(self, rolls, width, heuristic, limit):
        """ray-search.  We pick the best candidates with a heuristic.  Only consider feasible hexgrids.  Also it's gonna be stochastic.
        width is the "ray-width".  i.e. it's the size of the frontier.  It's the size of the population.  It's the size of the pool of candidates we have each round.
        heuristic is a fn that takes a grid and outputs a score: higher is better.  It'll probably be O(N)"""
        def generate_delta():
            #probs = {-1: 0.01, 0: 0.95, 1: 0.03, 2: 0.01}
            probs = {0:0.96, 1: 0.01}
            assert sum(probs.values()) == 1
            pitems = list(probs.items())
            r = random.random()
            cum = 0
            for i in pitems:
                cum += i[1]
                if cum >= r:
                    return i[0]

        self.lorde = [] #i'm listening to lorde
        adam = HexagonGrid() #adam from adam & eve
        adam.cluster_heuristic_score = heuristic(adam)
        #holy shit this code got so bad so fast 
        #but whatever i won't have to maintain this since the competition ends in 2 days so yolo
        self.lorde.append(adam)
        for i in range(limit):
            unculled_candidates = self.lorde[:]
            for hexgrid in self.lorde:
                for i in range(rolls):
                    new_hexgrid = HexagonGrid(from_hexgrid = hexgrid)
                    ##################################
                    row, col = random.choice(new_hexgrid.active_coords)
                    if not new_hexgrid.increment_hex(row, col, True):
                        continue
                    else:
                        unculled_candidates.append(new_hexgrid)

                        if new_hexgrid.sum > self.best_sum_of_all_time:
                            self.best_sum_of_all_time = new_hexgrid.sum
                            print(new_hexgrid.get_raw_score())
                            new_hexgrid.pretty_print()
                            print(new_hexgrid.serialize())
                            best = get_best(n)

                            if best is None or new_hexgrid.get_raw_score() > best[0]:
                                print("new all-time-best")
                                new_hexgrid.save()
                                print("saved!")
                            elif best is not None:
                                print("all-time-best is {} :(".format(best[0]))

                
                    ############################
                    #if new_hexgrid.is_feasible():
                    #    unculled_candidates.append(new_hexgrid)
            unculled_candidates.sort(key=heuristic, reverse=True)
            #print(heuristic(unculled_candidates[0]))
            #unculled_candidates[0].pretty_print()
            self.lorde = unculled_candidates[:width]

        return unculled_candidates[0]


def clustering_heuristic(hexgrid):
    ans = 0
    for r, c in hexgrid.active_coords:
        for dependents in range(1,8):
            ans += hexgrid.grid_neighbor_count[r][c][dependents]*hexgrid.grid[r][c]*dependents
            #if hexgrid.grid_neighbor_count[r][c][dependents] == 1 and dependents < hexgrid.grid[r][c]:
            #    ans += hexgrid.grid_neighbor_count[r][c][dependents]*dependents*2
            #    ans +=  hexgrid.grid[r][c]*dependents*6
            #else:
            #    ans += hexgrid.grid[r][c]*dependents
    return ans


#h = HexagonGrid()
#h.monte_carlo_backtrack_start(branches)

#import cProfile
#cProfile.run('h.monte_carlo_backtrack_start(10)')

r = RaySearchFeasible()
r.ray_search_feasible(rolls=20, width=10, heuristic=clustering_heuristic, limit=2000)

