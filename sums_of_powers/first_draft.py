import random
import math
ITS = 1000
def run(n):
   best = None
   argbest = None
   for i in range(ITS):
      base = random.randint(4,1000)
      options = set(range(1,base))
      params = []
      accum = 0
      target = pow(base, n)
      for i in range(len(options)):
         pick = random.sample(options, 1)[0]
         if accum + pow(pick, n) <= target:
            accum += pow(pick, n)
            params.append(pick)
            options.remove(pick)
      if best is None or 0 <= target - accum < best:
         best = target - accum
         argbest = '{}^{}=>{{{}}}'.format(base, n, ','.join(str(x) for x in params)) 
   # print('{}^{} = {}'.format(ans, n, ' + '.join('{}^{}'.format(x, n) for x in params)))
   # print('{}^{}=>{{{}}}'.format(ans, n, ','.join(str(x) for x in params)))
   # print(pow(ans, n) - summation)
   print(argbest)
   print(best)
run(16)
