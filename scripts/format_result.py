import json
import sys

for line in sys.stdin:
    d = json.loads(line.strip().replace("'", '"'))
    break

print(d)

print('''Ans EM:   {:.2%}
Ans F1:   {:.2%}
Sup EM:   {:.2%}
Sup F1:   {:.2%}
Joint EM: {:.2%}
Joint F1: {:.2%}'''.format(*tuple(d[k] for k in ['em', 'f1', 'sp_em', 'sp_f1', 'joint_em', 'joint_f1'])))
