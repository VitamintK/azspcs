try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
from multiprocessing import Pool
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
    def pretty_print(self):
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