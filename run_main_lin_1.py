from model import Regressor
from environment_lin import EnvironmentStateless
import torch
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from data_utils import Supervised, Unsupervised
from torch.utils.data import DataLoader
import copy
from itertools import product
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

#################################
optimizer = "RRM"

data_folder = "GiveMeSomeCredit"

assert len(sys.argv)>=4
[tran, eps, seed] = sys.argv[1:4]
eps = float(eps)
seed = int(seed)

trans = [[tran, {"eps": [eps]}]]

steps = 100
lr = 0.01
batch_size = 32
max_train_steps = 100
early_stop = 10

seeds = [seed]

device = "cpu"

result_folder = "results"
#################################

for ((tran, metas), seed) in list(product(trans, seeds)):
    keys = metas.keys()
    values = metas.values()
    all_values = list(product(*values))
    for values in all_values:
        meta = {key:val for key,val in zip(keys,values)}
        meta_name = ""
        for key,val in zip(keys,values):
            meta_name += f"{key} {val} , "
        meta_name = meta_name[:-3]

        exp_name = f"{optimizer}__{data_folder}__{tran}__{meta_name}__{seed}"
        exp_folder = f"{result_folder}/{optimizer}/{data_folder}/{tran}/{meta_name}/{seed}"
        if os.path.isdir(exp_folder):
            continue
        print("#"*50)
        print(exp_name)
        print("#"*50)

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        theta_diffs = []
        step_losses = []
        perf_risks = []
        perf_accs = []
        all_thetas = []

        device = torch.device('cuda' if torch.cuda.is_available() and device=="cuda" else 'cpu')
        print(f"training with {device}")

        env = EnvironmentStateless(tran, data_folder, **meta).to(device)
        n_examples, n_features = env.n_examples, env.n_features
        model = Regressor(n_features, [], bias=True).to(device)

        lossFunc = BCEWithLogitsLoss()

        for step in range(steps+1):
            X, Y = env.x, env.y
            if step>0:
                theta_diff = env.step(model)
                X, Y = env.x, env.y
                if step==1:
                    theta_diff = None
                
                #########################################
                print(f"{step}: diff - {theta_diff}, accuracy - {perf_accs[-1]}, risk - {perf_risks[-1]}")
                if theta_diff is None:
                    theta_diffs.append(-1)
                else:
                    theta_diffs.append(theta_diff)
                step_losses.append(step_loss)
                #########################################
            
            ##################################################################################
            dataset = Supervised(X, Y, n_examples)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            best_model = None
            best_ind = None
            best_loss = None
            opt = optim.Adam(model.parameters(), lr=lr)
            for i in range(max_train_steps):
                epoch_total_loss = 0
                epoch_loss = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    batch_len = x.shape[0]
                    preds = model(x)
                    loss = lossFunc(preds, y)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    epoch_loss += batch_len*loss.item()
                epoch_loss /= n_examples
                if best_loss is None or best_loss>epoch_loss:
                    best_ind = i
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    continue
                if i>best_ind+early_stop:
                    break
            model = best_model
            step_loss = best_loss
            ##################################################################################

            with torch.no_grad():
                X, Y = env.peek(model)
                dataset = Supervised(X, Y, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                perf_risk = 0
                perf_acc = 0
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    batch_len = x.shape[0]
                    preds = model(x)
                    loss = lossFunc(preds, y)
                    perf_risk += batch_len*loss.item()
                    perf_acc += ((preds>0)==y).sum().item()
                perf_risk /= n_examples
                perf_acc /= n_examples

                perf_risks.append(perf_risk)
                perf_accs.append(perf_acc)
            
            theta = torch.zeros((n_features+1,))
            EnvironmentStateless._update_theta(model, theta)
            all_thetas.append(theta.numpy())


        os.makedirs(exp_folder, exist_ok=True)
        with open(f"{exp_folder}/theta_diffs.txt", 'w+') as f:
            for theta_diff in theta_diffs:
                f.write(f"{theta_diff}\n")
        with open(f"{exp_folder}/step_losses.txt", 'w+') as f:
            for step_loss in step_losses:
                f.write(f"{step_loss}\n")
        with open(f"{exp_folder}/perf_risks.txt", 'w+') as f:
            for perf_risk in perf_risks:
                f.write(f"{perf_risk}\n")
        with open(f"{exp_folder}/perf_accs.txt", 'w+') as f:
            for perf_acc in perf_accs:
                f.write(f"{perf_acc}\n")
        with open(f"{exp_folder}/all_thetas.npy", 'wb+') as f:
            all_thetas = np.stack(all_thetas)
            np.save(f, all_thetas)
