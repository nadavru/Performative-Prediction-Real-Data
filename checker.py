import os
from itertools import product
import shutil

#################################
optimizers = ["RRM", "RRM + ADAM (0.1)", "RRM + ADAM (0.1) + lookahead", "RRM + ADAM (0.1) + learned lookahead", "RRM + ADAM (0.9)", "RRM + ADAM (0.9) + lookahead", "RRM + ADAM (0.9) + learned lookahead"]
data_folder = "GiveMeSomeCredit"
trans = [["tran1", {"eps": [0.01, 1, 100, 1000, -0.01, -1, -100, -1000]}], ["tran2", {"eps": [0.01, 1, 100, 1000, -0.01, -1, -100, -1000]}]]
seeds = list(range(1234,1244))

result_folder = "results"

files_to_check = ["theta_diffs.txt", "step_losses.txt", "perf_accs.txt", "perf_risks.txt"]
#################################

currents = {(optimizer, tran): 0 for optimizer, (tran, metas) in list(product(optimizers, trans))}
totals = {(optimizer, tran): 0 for optimizer, (tran, metas) in list(product(optimizers, trans))}

print("All Problems:")
print("#"*50)
total = 0
count = 0
for optimizer in optimizers:
    for ((tran, metas), seed) in list(product(trans, seeds)):
        keys = metas.keys()
        values = metas.values()
        all_values = list(product(*values))
        for values in all_values:
            meta = {key:val for key,val in zip(keys,values)}
            meta_name = ""
            for key,val in zip(keys,values):
                meta_name += f"{key} {float(val)} , "
            meta_name = meta_name[:-3]

            exp_name = f"{optimizer}__{data_folder}__{tran}__{meta_name}__{seed}"
            
            exp_folder = f"{result_folder}/{optimizer}/{data_folder}/{tran}/{meta_name}/{seed}"
            
            problem = False
            if not os.path.isdir(exp_folder):
                problem = True
            for file_to_check in files_to_check:
                if not os.path.isfile(f"{exp_folder}/{file_to_check}"):
                    problem = True
            
            if problem:
                print(exp_name)
            else:
                count += 1
                currents[(optimizer, tran)] += 1
            
            total += 1
            totals[(optimizer, tran)] += 1
print("#"*50)
for key in currents.keys():
    print(f"{key}: {currents[key]}/{totals[key]}")
print("#"*50)
print(f"Progress: {count}/{total}")
print("All Done!!!")
