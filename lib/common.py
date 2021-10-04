try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
from multiprocessing import Pool
import random
PARALLELIZATION = 4

def f(args):
    obj, temperature = args
    return obj.sample_neighbor(temperature)

class Solution:
    def sample_neighbors(self, n: int, temperature):
        # with Pool(PARALLELIZATION) as p:
        #     res = p.map(f, [(self, temperature) for i in range(n)])
        # return res
        return (self.sample_neighbor(temperature) for i in range(n)) 
    def sample_neighbor(self, temperature):
        """temperature is a float from 0 to 1.  1 meaning lots of change, 0 being no change."""
        raise NotImplementedError 
    def heuristic(self):
        # heuristic score to use when evaluating whether a solution is more promising or not to explore
        raise NotImplementedError 
    def score(self):
        # formal score in azspcs for this solution
        raise NotImplementedError 
    # def is_feasible(self):
    #     raise NotImplementedError 
    # def is_potentially_feasible(self):
    #     raise NotImplementedError
    def serialize(self):
        raise NotImplementedError
    def pretty(self):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError

class MonteCarloBeamSearcher:
    def __init__(self, start, best_of_all_time):
        self.solution = start
        self.best_of_all_time = best_of_all_time
    def go(self, iterations = 1000, population = 10, samples = 20):
        # linear temperature function:
        temperature_function = lambda x: (1 - x/iterations )
        if self.solution is None:
            raise ValueError
        candidates = [self.solution]
        ans = self.solution
        for it in tqdm(range(iterations)):
            if it%100 == 0:
                for cand in candidates:
                    print(cand.score())
                    print(cand.pretty())
            next_candidates = [candidates[0]]
            for cand in candidates:
                next_candidates.extend(cand.sample_neighbors(samples, temperature=temperature_function(it)))
            b = max(*next_candidates, key = lambda x: x.score())
            bscore = b.score()
            if bscore > ans.score():
                print(bscore)
                # print(b)
                # print(b.serialize())
                ans = b
            if bscore > self.best_of_all_time:
                b.save()
                self.best_of_all_time = bscore
            else:
                pass
                # print(ans.score(), bscore)
            next_candidates.sort(key=lambda x: x.heuristic(), reverse=True)
            candidates = next_candidates[:population]
        return ans

import sys
sys.setrecursionlimit(100000)
class ExhaustiveBacktracker:
    def __init__(self, start: Solution, best_of_all_time, with_cache=False):
        self.solution = start
        self.best_of_all_time = best_of_all_time
        self.its = 0
        self.with_cache = with_cache
        if with_cache:
            self.cache = set()
    def go(self):
        if self.with_cache:
            if self.solution._hash in self.cache:
                print('hey!')
                return
            self.cache.add(self.solution._hash)
        if len(self.solution._taboo_free) + self.solution.score() <= self.best_of_all_time:
            # print(len(self.solution.active_coord_set))
            # print(self.solution.score())
            # print(len(self.solution._taboo_free), self.solution._taboo_free)
            # print(self.solution.pretty())
            return -1
        self.its += 1
        if self.its % 5000 == 0:
            print(f'{self.best_of_all_time=}')
            print(f'{self.its=}')
            print(f'{self.solution.score()=}')
            print(len(self.solution._taboo_free), self.solution._taboo_free)
            print(self.solution.pretty())
        # all_actions = list(sorted(self.solution.get_all_actions())) #SORTING ONLY FOR DEBUGGING.  DO NOT NEED THE SORTING FOR ACTUAL RUNNING
        all_actions = list(self.solution.get_all_actions())
        random.shuffle(all_actions)
        added_actions = []
        for action in all_actions:
            if len(self.solution._taboo_free) + self.solution.score() <= self.best_of_all_time:
                break
            self.solution.apply_action(action)
            score = self.solution.score()
            if score > self.best_of_all_time:
                self.solution.save()
                self.best_of_all_time = score
            self.go()
            self.solution.undo_action(action)
            self.solution._add_taboo(action)
            added_actions.append(action)
        for action in added_actions:
            self.solution._remove_taboo(action)

class SamplingBacktracker:
    def __init__(self, start: Solution, best_of_all_time, k, with_cache=False):
        self.solution = start
        self.best_of_all_time = best_of_all_time
        self.k = k
        self.its = 0
        self.with_cache = with_cache
        if with_cache:
            self.cache = set()
    def go(self):
        if self.with_cache:
            if self.solution._hash in self.cache:
                print('hey!')
                return
            self.cache.add(self.solution._hash)
        if len(self.solution._taboo_free) + self.solution.score() <= self.best_of_all_time:
            # print(len(self.solution.active_coord_set))
            # print(self.solution.score())
            # print(len(self.solution._taboo_free), self.solution._taboo_free)
            # print(self.solution.pretty())
            return -1
        self.its += 1
        if self.its % 100 == 0:
            print(f'{self.best_of_all_time=}')
            print(f'{self.its=}')
            print(f'{self.solution.score()=}')
            print(len(self.solution._taboo_free), self.solution._taboo_free)
            print(self.solution.pretty())
        # all_actions = list(sorted(self.solution.get_all_actions())) #SORTING ONLY FOR DEBUGGING.  DO NOT NEED THE SORTING FOR ACTUAL RUNNING
        all_actions = random.sample(list(self.solution.get_all_actions()), k=min(self.k, len(self.solution.get_all_actions())))
        all_actions_ranked = []
        for action in all_actions:
            if len(self.solution._taboo_free) + self.solution.score() <= self.best_of_all_time:
                break
            self.solution.apply_action(action)
            score = self.solution.score()
            if score > self.best_of_all_time:
                self.solution.save()
                self.best_of_all_time = score
            all_actions_ranked.append((self.solution.heuristic(), action))
            self.solution.undo_action(action)
        all_actions_ranked.sort(reverse=True)
        for heuristic, action in all_actions_ranked:
            if len(self.solution._taboo_free) + self.solution.score() <= self.best_of_all_time:
                break
            self.solution.apply_action(action)
            score = self.solution.score()
            if score > self.best_of_all_time:
                self.solution.save()
                self.best_of_all_time = score
            self.go()
            self.solution.undo_action(action)
        