import numpy as np
import random

# python probably the worst possible language for this.
# TODO (never gonna do this): use more of np for speedup

class Solution:
    def sample_neighbors(self, n: int):
        return (self.sample_neighbor() for i in range(n)) 
    def sample_neighbor(self):
        raise NotImplementedError 
    def heuristic(self):
        # heuristic score to use when evaluating whether a solution is more promising or not to explore
        raise NotImplementedError 
    def score(self):
        # formal score in azspcs for this solution
        raise NotImplementedError 
    def is_feasible(self):
        raise NotImplementedError 
    def is_potentially_feasible(self):
        raise NotImplementedError 

class Solver:
    def __init__(self, start = None):
        self.solution = start
    def monte_carlo_beam_search(self, iterations = 1000, population = 10, samples = 20):
        """keeps a list of best candidate solutions (of size `population`).  Each iteration,
        samples `samples` neighbors of each candidate solution, then keeps the best-scoring `population`.
        """
        if self.solution is None:
            raise ValueError
        candidates = [self.solution]
        ans = self.solution
        for it in range(iterations):
            print(it)
            next_candidates = [candidates[0]]
            for cand in candidates:
                next_candidates.extend(cand.sample_neighbors(samples))
            b = max(*next_candidates, key = lambda x: x.score())
            if b.score() > ans.score():
                print(b.score())
                print(b)
                print(b.serialize())
                ans = b
            next_candidates.sort(key=lambda x: x.heuristic(), reverse=True)
            candidates = next_candidates[:population]
        return ans
    def monte_carlo_backtrack(self):
        # soln
        # for i in whatever:
        #   action = soln.sample_mutation() <- like an action in react
        #   soln.apply_mutation(action)
        #   recurse
        #   soln.rollback_mutation(action)
        pass


########## general framework above ############
########## nearness reversed below ############

class Nearness(Solution):
    original_coords = dict()
    alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234'
    lower_bounds = [None, 0, 10, 72, 816, 3800, 16902, 52528, 155840, 381672, 902550, 1883244, 3813912, 7103408, 12958148, 22225500, 37474816, 60291180, 95730984, 146469252, 221736200, 325763172, 474261920, 673706892, 949783680, 1311600000, 1799572164, 2425939956, 3252444776, 4294801980, 5643997650]
    def __init__(self, n, board = None, score = None):
        self.n = n
        if board is not None:
            self.board = board
        else:
            self.board = [[(c,r) for c in range(n)] for r in range(n)]
        self._score = score
    def sample_neighbor(self):
        board = [[i for i in r] for r in self.board] #replace with np.clone
        a, b, c, d = (random.randrange(0, self.n) for i in range(4))
        board[a][b], board[c][d] = board[c][d], board[a][b]
        a, b, c, d = (random.randrange(0, self.n) for i in range(4))
        board[a][b], board[c][d] = board[c][d], board[a][b]
        a, b, c, d = (random.randrange(0, self.n) for i in range(4))
        board[a][b], board[c][d] = board[c][d], board[a][b]
        a, b, c, d = (random.randrange(0, self.n) for i in range(4))
        board[a][b], board[c][d] = board[c][d], board[a][b]
        return Nearness(self.n, board)
    def score(self):
        # TODO: look into using https://docs.scipy.org/doc/numpy/reference/arrays.nditer.html for iteration
        ans = 0
        for r1 in range(self.n):
            for c1 in range(self.n):
                for r2 in range(self.n):
                    for c2 in range(self.n):
                        ans += Nearness.distance(self.n, (r1, c1), (r2, c2)) * Nearness.original_distance(self.n, self.board[r1][c1], self.board[r2][c2])
        # we double-count each pair, so divide by 2
        return -ans//2 + Nearness.lower_bounds[self.n]
    def heuristic(self):
        return self.score()
    def swap(self, coords1, coords2):
        pass
        # a,b = coords1
        # c,d = coords2
        # board[a][b], board[c][d] = board[c][d], board[a][b]
        # if self._score is not None:
        #     self._score
    def pairs(n):
        # TODO: no need to compute this every time
        pass
    def distance(n, x1, x2):
        r1, c1 = x1
        r2, c2 = x2
        h1 = abs(c1 - c2)
        h2 = n - h1
        v1 = abs(r1 - r2)
        v2 = n - v1
        return min(h1, h2) * min(h1, h2) + min(v1, v2) * min(v1, v2)
    def original_distance(n, og_coords1, og_coords2):
        return Nearness.distance(n, og_coords1, og_coords2)
    def serialize(self):
        return ',\n'.join('('+','.join(Nearness.alph[x]+Nearness.alph[y] for x,y in r)+')' for r in self.board)
    def pretty_str(self):
        ans = ''
        for r in self.board:
            ans += ' '.join(Nearness.alph[x]+Nearness.alph[y] for x,y in r) + '\n'
        return ans
    def __str__(self):
        return self.pretty_str()

def main():
    n = Nearness(n = 6)
    print(n.board)
    print(n.score())
    print(n)
    s = Solver(n)
    ans = s.monte_carlo_beam_search(1000, 10, 2)
    print(ans)
    print(ans.score())
    print(ans.serialize())


main()