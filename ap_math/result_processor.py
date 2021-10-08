import argparse
DIRECTORY = 'results'

ALL_NS = [2, 6, 11, 18, 27, 38, 50, 65, 81, 98, 118, 139, 162, 187, 214, 242, 273, 305, 338, 374, 411, 450, 491, 534, 578]
def get_values(ns=ALL_NS):
    ans = []
    scores = []
    for val in ns:
        try:
            with open(f'{DIRECTORY}/{val}_repr.out', 'r') as f:
                ans.append(f.read())
        except FileNotFoundError:
            pass
        try:
            with open(f'{DIRECTORY}/{val}_score.out', 'r') as f:
                # scores.append(f'{val}: {f.read()}')
                scores.append(int(f.read()))
        except FileNotFoundError:
            pass
    return ans, scores

def print_scores(ns, scores):
    for n,score in zip(ns, scores):
        print(f'{n}: {score}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ns', type=int, nargs="*", help='ns')
    args = parser.parse_args()

    if args.ns == []:
        values = ALL_NS
    else:
        values = args.ns
    ans, scores = get_values(values)
    print(';'.join(ans))
    print_scores(scores)
    # print('\n'.join(scores))