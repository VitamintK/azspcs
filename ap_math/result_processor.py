import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ns', type=int, nargs="*", help='ns')
args = parser.parse_args()

directory = 'results'
if args.ns == []:
    values = [2, 6, 11, 18, 27, 38, 50, 65, 81, 98, 118, 139, 162, 187, 214, 242, 273, 305, 338, 374, 411, 450, 491, 534, 578]
else:
    values = args.ns
ans = []
scores = []
for val in values:
    try:
        with open(f'{directory}/{val}_repr.out', 'r') as f:
            ans.append(f.read())
    except FileNotFoundError:
        pass
    try:
        with open(f'{directory}/{val}_score.out', 'r') as f:
            scores.append(f'{val}: {f.read()}')
    except FileNotFoundError:
        pass
print(';'.join(ans))
print('\n'.join(scores))