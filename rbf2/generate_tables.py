import os
import csv
import numpy as np
from collections import defaultdict

# Read summary
summary_data = defaultdict(dict)
with open('/home/pedro/cefet/projeto_perceptron/rbf2/results/summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        net = row['Topology']
        t = row['Training']
        summary_data[net][t] = {'EQM': float(row['EQM']), 'Epocas': int(row['Epocas'])}

# Table 1:
nets = ['5', '10', '15']
trainings = ['T1', 'T2', 'T3']
print("=== TABLE 1 ===")
for i, t in enumerate(trainings, 1):
    row = [f"{i}o ({t})"]
    for net in nets:
        row.append(f"{summary_data[net][t]['EQM']:.6f}")
        row.append(str(summary_data[net][t]['Epocas']))
    print(" | ".join(row))

# Read predictions
preds = defaultdict(lambda: defaultdict(dict))
with open('/home/pedro/cefet/projeto_perceptron/rbf2/evaluation/predictions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        net = row['Rede']
        t = row['Treinamento']
        sample = int(row['Amostra'])
        preds[sample][net][t] = float(row['y_pred'])

# Read test samples
test_samples = [
    [0.5102, 0.7464, 0.0860, 0.5965],
    [0.8401, 0.4490, 0.2719, 0.6790],
    [0.1283, 0.1882, 0.7253, 0.4662],
    [0.2299, 0.1524, 0.7353, 0.5012],
    [0.3209, 0.6229, 0.5233, 0.6810],
    [0.8203, 0.0682, 0.4260, 0.5643],
    [0.3471, 0.8889, 0.1564, 0.5875],
    [0.5762, 0.8292, 0.4116, 0.7853],
    [0.9053, 0.6245, 0.5264, 0.8506],
    [0.8149, 0.0396, 0.6227, 0.6165],
    [0.1016, 0.6382, 0.3173, 0.4957],
    [0.9108, 0.2139, 0.4641, 0.6625],
    [0.2245, 0.0971, 0.6136, 0.4402],
    [0.6423, 0.3229, 0.8567, 0.7663],
    [0.5252, 0.6529, 0.5729, 0.7893],
]

print("\n=== TABLE 2 ===")
for i, row in enumerate(test_samples, 1):
    out_row = [f"{i:02d}", f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}"]
    for net in nets:
        for t in trainings:
            out_row.append(f"{preds[i][net][t]:.4f}")
    print(" | ".join(out_row))

# Read errors from evaluation.md
print("\n=== ERRORS ===")
errors = {}
with open('/home/pedro/cefet/projeto_perceptron/rbf2/evaluation/evaluation.md', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('| 5 |') or line.startswith('| 10 |') or line.startswith('| 15 |'):
            parts = [p.strip() for p in line.split('|')]
            net = parts[1]
            t = parts[2]
            eqm = parts[3]
            var = parts[4]
            errors[(net, t)] = (eqm, var)

err_row = ["Erro Relativo Médio (%)"]
var_row = ["Variância (%)"]
for net in nets:
    for t in trainings:
        err_row.append(errors[(net, t)][0])
        var_row.append(errors[(net, t)][1])
print(" | ".join(err_row))
print(" | ".join(var_row))

