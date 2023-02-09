from model import Regressor, PolynomV2
from environment_lin import EnvironmentStateless
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.optim as optim
from data_utils import Supervised, Unsupervised
from torch.utils.data import DataLoader
import copy
from itertools import product
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from torch import nn

#################################
optimizer = "RRM + ADAM (0.9) + learned lookahead"

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
adam_lr = 0.9
beta1 = 0.9
beta2 = 0.999
eps = 10**-8
lookahead_train_steps = 1

T_degree = 1
T_lr = 0.01
T_iterations_until_lookahead = 5

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
        
        T_operator = PolynomV2(n_features+1+n_features, T_degree, n_output=n_features+1).to(device)

        lossFunc = BCEWithLogitsLoss()
        T_lossFunc = MSELoss()
        
        mt, vt = 0, 0

        for step in range(steps+1):
            X, Y = env.x, env.y
            if step>0:
                theta_diff = env.step(model)
                X, Y = env.x, env.y
                if step==1:
                    theta_diff = None
                
                theta = torch.zeros((n_features,)).to(device)
                EnvironmentStateless._update_theta(model, theta)

                dataset = Supervised(torch.cat((env.x_src, env.y_src), 1), torch.cat((X,Y), 1), n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                best_model = None
                best_ind = None
                best_loss = None
                opt_T = optim.Adam(T_operator.parameters(), lr=T_lr)
                base_model = copy.deepcopy(T_operator)
                for i in range(max_train_steps):
                    epoch_loss = 0
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]
                        x_cat = torch.cat((x,theta.repeat(batch_len,1)), 1)
                        preds = T_operator(x_cat)
                        loss = T_lossFunc(preds, y)

                        opt_T.zero_grad()
                        loss.backward()
                        opt_T.step()
                        epoch_loss += batch_len*loss.item()
                    epoch_loss /= n_examples
                    if best_loss is None or best_loss>epoch_loss:
                        best_ind = i
                        best_loss = epoch_loss
                        best_model = copy.deepcopy(T_operator)
                        continue
                    if i>best_ind+early_stop:
                        break
                T_operator = best_model
                T_loss = best_loss
                
                #########################################
                print(f"{step}: diff - {theta_diff}, accuracy - {perf_accs[-1]}, risk - {perf_risks[-1]}, T-loss - {T_loss:.5f}")
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
            
            with torch.no_grad():
                base_theta = torch.zeros((n_features+1,))
                EnvironmentStateless._update_theta(model, base_theta)
            
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
            
            if step>=T_iterations_until_lookahead:
                adam_step = step-(T_iterations_until_lookahead-1)
                with torch.no_grad():
                    theta = torch.zeros((n_features+1,))
                    EnvironmentStateless._update_theta(model, theta)
                    model_grad = base_theta-theta
                    
                    mt = beta1*mt + (1-beta1)*model_grad
                    vt = beta2*vt + (1-beta2)*(model_grad**2)

                    mt_ = mt/(1-beta1**adam_step)
                    vt_ = vt/(1-beta2**adam_step)

                    base_theta -= adam_lr*mt_/(vt_**0.5+eps)
                    
                    def init_weights(m):
                        if isinstance(m, nn.Linear):
                            m.weight = nn.Parameter(base_theta[None,:n_features])
                            m.bias = nn.Parameter(base_theta[None,-1])
                    model.block.apply(init_weights)
                    
                dataset = Supervised(env.x_src, env.y_src, n_examples)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                opt = optim.Adam(model.parameters(), lr=lr)
                T_operator.eval()
                for i in range(lookahead_train_steps):
                    epoch_loss = 0
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        batch_len = x.shape[0]

                        eye = torch.eye(n_features).to(device)
                        bias = model(torch.zeros(n_features).to(device))
                        params = model(eye).view(1,-1)-bias

                        x_cat = torch.cat((x,y,params.repeat(batch_len,1)), 1)
                        y_cat = T_operator(x_cat)
                        x_pred = y_cat[:,:n_features]
                        y_pred = y_cat[:,n_features:]

                        preds = model(x_pred)
                        #loss = lossFunc(preds, y_pred)
                        loss = lossFunc(preds, y)
                        
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        epoch_loss += batch_len*loss.item()
                    epoch_loss /= n_examples
                T_operator.train()
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
