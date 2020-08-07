import math
import random
import cProfile

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
    def serialize(self):
        raise NotImplementedError

class MonteCarloBeamSearcher:
    def __init__(self, start):
        self.solution = start
    def go(self, iterations = 1000, population = 10, samples = 20):
        if self.solution is None:
            raise ValueError
        candidates = [self.solution]
        ans = self.solution
        for it in range(iterations):
            if it%1000 == 0:
                print('+++++++++++', it)
                for cand in candidates:
                    targ, have = pow(cand.base,cand.n), sum(pow(p,cand.n) for p in cand.params)
                    print(cand.serialize(), cand.score(), targ, have, targ-have)
            next_candidates = [candidates[0]]
            for cand in candidates:
                next_candidates.extend(cand.sample_neighbors(samples))
            b = max(*next_candidates, key = lambda x: x.score())
            if b.score() > ans.score():
                print(b.score())
                # print(b)
                print(b.serialize())
                ans = b
            next_candidates.sort(key=lambda x: x.heuristic(), reverse=True)
            candidates = next_candidates[:population]
        return ans

#############################################################################

class Powers(Solution):
    cache = dict()
    def __init__(self, n, base=None, params=None):
        self.n = n
        self.params = set() if params is None else params
        self.base = random.randint(50,200) if base is None else base# idk just some random shit
        self.e = abs(Powers.pow(self.base, n) - sum(Powers.pow(p, n) for p in self.params))
    def greedy_optimize(self):
        # maybe not the best API, probably because Solver/Solution abstraction splitup is not perfect
        # mutates self until in a local minimum
        pass
    def sample_neighbor(self):
        # types of mutations
        # deletion {3,5,6} -> {3,6}
        # increment/decrement {3,5,6} -> {3,5,7}
        # insertion {3,5,6} -> {3,4,5,6}
        new_ps = set()
        for p in self.params:
            # deletion
            if random.random() < 0.04:
                continue
            # incre/decre-ment
            if random.random() < 0.04:
                if random.random() < 0.5:
                    new_ps.add(p+1)
                else:
                    if p > 1:
                        new_ps.add(p-1)
                continue
            new_ps.add(p)
        for i in range(int(random.normalvariate(3.5,2))):
            new_ps.add(random.randrange(self.base))
        return Powers(self.n, self.base, new_ps)
    def serialize(self):
        return '{}^{}=>{{{}}}'.format(self.base, self.n, ','.join(str(x) for x in self.params))
    def score(self):
        return -1 * (math.log(self.get_e() + 1) + 1)
    def heuristic(self):
        return self.score()
    def get_e(self):
        return self.e
    def pow(base, n):
        if (base, n) in Powers.cache:
            return Powers.cache[(base,n)]
        else:
            print("not using cache")
            ans = pow(base, n)
            Powers.cache[(base,n)] = ans
            return ans




def main():
    # 99^16=>{64,96,72,92,48,18,19,84,53,24,25,58,27,28}
    initial_solution = Powers(16)
    s = MonteCarloBeamSearcher(initial_solution)
    ans = s.go(11234567, 16, 16)
    print(ans)
    print(ans.score())
    print(ans.serialize())

# EXP = 16
# def is_too_small_or_equal(mid)
# def main2():
#     base = int(input())
#     target = 
#     r = pow(2, base)
#     l = 0
#     while r - 1 < l:
#         mid = (r+l)//2
#         if is_too_small_or_equal(mid):
#             l = mid
#         else:
#             r = mid

DEBUG = 0

def debug(*args):
    if DEBUG > 0:
        print(*args)

def bitset_to_set(bitset):
    s = set()
    k = 0
    while bitset != 0:
        if bitset&1:
            s.add(k)
        bitset >>= 1
        k += 1
    return s

def set_to_bitset(s):
    bitset = 0
    for e in s:
        bitset |= (1 << e)
    return bitset

def fs(x):
    return frozenset(x)

def sf(x):
    return set(x)

def main2(targ_base = None, params=None):
    # find a top base, B
    # then, exhaustively search all permutations of a1^exp + a2^exp ... 
    # where all `a`s must be at least B-15 or so.
    # find the closest one to TARG, then repeat for B-16 to B-30, etc.
    # fuuuuuuuuuuuuuuck this problem is so hard :(
    p = 16
    window_size = 10
    window_slide = 2
    beam_size = 50
    if targ_base is None:
        targ_base = random.randint(100,200)
    targ = pow(targ_base, p)
    if params is not None:
        candidates = [set(params)]
    else:
        candidates = [set(x for x in range(1,targ_base) if random.random() > 0.5)]
    # soln = Powers(p, targ, params)
    pows = []
    for i in range(targ_base+1):
        pows.append(pow(i, p))
    # def get_possibilities(EXP, TERMS, IGNORE_SMALLEST=0):
    #     possibilities = []
        
    #     # for i in range(TERMS+1):
    #     #     pows.append(pow(i,EXP))
    #     for i in range(pow(2,TERMS-IGNORE_SMALLEST)):
    #         x = i
    #         b = 0
    #         ans = 0
    #         j = IGNORE_SMALLEST
    #         while x != 0:
    #             if x&1:
    #                 ans += pows[j]
    #             x >>= 1
    #             j += 1
    #         possibilities.append((i, ans))
        # return possibilities
    debug("starting with")
    debug(p, window_size, targ_base, targ, candidates)
    for i in range(targ_base-1, 1, -window_slide):
        possibilities = set()
        for params in candidates:
            existing_sum = sum(pows[x] for x in params if not (i-window_size < x <= i))
            # debug([x for x in params if not (i-window_size < x <= i)], existing_sum)
            eff_window_size = min(window_size, i)
            for j in range(pow(2, eff_window_size)):
                x = j
                k = i - eff_window_size + 1
                new_params = set(params)
                ans = 0
                for offset in range(eff_window_size):
                    if x&1:
                        new_params.add(i-eff_window_size+1+offset)
                        ans += pows[k]
                    else:
                        new_params.discard(i-eff_window_size+1+offset)
                    x >>= 1
                    k += 1
                possibilities.add((set_to_bitset(new_params), abs(existing_sum + ans - targ)))
        possibilities = list(possibilities)
        possibilities.sort(key = lambda x: x[1])
        # debug('possibilities', possibilities[:50])
        # best = possibilities[0][0]
        # new_candidates = []
        # for cand in possibilities:
        #     new_params = set(params)
        #     for j in range(window_size):
        #         if best&1:
        #             debug("adding", i-window_size+1+j)
        #             new_params.add(i-window_size+1+j)
        #         else:
        #             debug("removing", i-window_size+1+j)
        #             new_params.discard(i-window_size+1+j)
        #         best>>=1
        #     new_candidates.append(new_params)
        # candidates = new_candidates
        candidates = [bitset_to_set(x[0]) for x in possibilities[:beam_size]]
        debug("window from {} to {}".format(i-window_size+1, i))
        for cand in candidates:
            debug(cand)

        # debug(params)
        # debug("{}^{} = {}, sum = {}".format(targ_base, p, targ, sum(pows[x] for x in params)))
    params = candidates[0]
    debug(params)
    score = 1 + math.log(1+ abs(sum(pows[x] for x in params) - targ)) 
    debug(score)
    if score < 30:
        print('$$$')
        print(score)
        print(params)
        print("{}^{} = {}, sum = {}".format(targ_base, p, targ, sum(pows[x] for x in params)))
    debug("{}^{} = {}, sum = {}".format(targ_base, p, targ, sum(pows[x] for x in params)))





# for i in range(2,1000):
#     main2(i)
# main2(102, {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 20, 24, 28, 29, 30, 33, 34, 35, 38, 43, 44, 46, 49, 51, 52, 55, 56, 58, 61, 62, 63, 68, 71, 72, 74, 76, 77, 80, 83, 88, 89, 95, 96})
cProfile.run("""for i in range(2):
    main2()""")

