"""
Code adapted from https://github.com/yegcjs/mixinglaws.
"""

# import multiprocessing as mp
import multiprocessing as mp
import torch
from functools import partial
import logging
import numpy as np
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import time

try:
    mp.set_start_method("fork")
except RuntimeError:
    pass


def calculate_r_squared(actuals, predictions):
    actuals, predictions = actuals.numpy(), predictions.numpy()
    # Calculate the total sum of squares
    total_sum_of_squares = np.sum((actuals - np.mean(actuals)) ** 2)
    # Calculate the residual sum of squares
    residual_sum_of_squares = np.sum((actuals - predictions) ** 2)
    # Calculate R-squared
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


def fit_scaling_laws(func, valid_split, x, y, max_step, eps, delta, init_param):
    param = torch.nn.Parameter(init_param)
    x, y = torch.tensor(x).to(param), torch.tensor(y).to(param)
    if valid_split == 0:
        train_x, eval_x = x, x[:0]
        train_y, eval_y = y, y[:0]
    else:
        train_x, eval_x = x[:-valid_split], x[-valid_split:]
        train_y, eval_y = y[:-valid_split], y[-valid_split:]
    optimizer = torch.optim.LBFGS([param], lr=0.01, history_size=10, max_iter=20, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.AdamW([param], lr=1e-3)
    def closure():
        optimizer.zero_grad()
        prediction = func(train_x, param)
        loss = torch.nn.functional.huber_loss(train_y, prediction, delta=delta, reduction="sum")
        loss.backward()
        return loss
    
    min_loss, best_param = 1e10, None
    best_step = 0
    for i in range(max_step):
        loss = optimizer.step(closure).item()
        # prediction = func(train_x, param) 
        # train_r2 = calculate_r_squared(train_y, prediction)
        with torch.no_grad():
            if len(eval_x) > 1:   
                eval_prediction = func(eval_x, param)
                eval_loss = torch.nn.functional.huber_loss(eval_prediction, eval_y, delta=delta).item() 
                # eval_r2 = calculate_r_squared(eval_y, eval_prediction)
                # eval_loss = -eval_r2
            elif len(eval_x) == 1:
                eval_prediction = func(eval_x, param)
                eval_loss = torch.nn.functional.mse_loss(eval_prediction, eval_y).item()
            else:
                eval_prediction = func(train_x, param)
                eval_loss = torch.nn.functional.huber_loss(eval_prediction, train_y, delta=delta).item() 
                # eval_loss = -calculate_r_squared(train_y, eval_prediction)
                # eval_loss = -eval_r2
        if eval_loss <= min_loss: # FIXME
            min_loss = eval_loss
            best_param = param.detach().clone()
            best_step = i
        # print(loss)
        if np.abs(min_loss - eval_loss) < eps:
            assert False
            break

    if best_param is None:
        print(min_loss, eval_loss)
        print(eval_prediction)
        print(train_x, param)

    return min_loss, best_param.detach().cpu().numpy(), best_step





class ScalingLaw:
    def __init__(self, func):
        self.func = func
        self.params = None
        
    def fit(self, x, y, init_params, max_step=20, eps=0, workers=-1, valid_split=0, delta=0.01):
        if workers == -1:
            workers = min(8, mp.cpu_count())
        init_params = [torch.tensor(init_param, dtype=torch.float32) for init_param in init_params]
        minloss, optimal_param = 1e10, None
        _fit = partial(fit_scaling_laws, self.func, valid_split, x, y, max_step, eps, delta)
        if workers != 1:
            best_step = 0
            with mp.Pool(workers, maxtasksperchild=20) as p:
                for loss, param, step in tqdm(p.imap_unordered(_fit, init_params, chunksize=2), total=len(init_params)):
                    if loss < minloss:
                        minloss = loss
                        optimal_param = param
                        best_step = step
        else:
            for init_param in tqdm(init_params):
                loss, param, step = _fit(init_param)
                # print(param)
                if loss < minloss:
                    minloss = loss
                    optimal_param = param 
        self.params = optimal_param.tolist()
        print(f"min loss: {minloss}")
        return self.params
    

def fit_multi_obj_scaling_laws(funcs, valid_split, x, ys, max_step, eps, delta, loss_type, init_param):
    # requirements: each function in funcs must take in the same init_param. This may mean that some init_params are not used in all funcs.
    param = torch.nn.Parameter(init_param)
    x, ys = torch.tensor(x).to(param), torch.tensor(ys).to(param)
    if valid_split == 0:
        train_x, eval_x = x, x[:0]
        train_y, eval_y = ys, ys[:0]
    else:
        train_x, eval_x = x[:-valid_split], x[-valid_split:]
        train_y, eval_y = ys[:-valid_split], ys[-valid_split:]
    optimizer = torch.optim.LBFGS([param], lr=0.01, history_size=10, max_iter=20, line_search_fn="strong_wolfe")
    # optimizer = torch.optim.AdamW([param], lr=1e-3)
    def closure():
        optimizer.zero_grad()
        loss = 0
        for func, y in zip(funcs, train_y):
            prediction = func(train_x, param)

            if loss_type == "huber":
                loss += torch.nn.functional.huber_loss(y, prediction, delta=delta, reduction="sum")
            elif loss_type == "mse":
                loss += torch.nn.functional.mse_loss(y, prediction, reduction="sum")
        loss.backward()
        return loss
    
    min_loss, best_param = 1e10, None
    best_step = 0
    for _ in range(max_step):
        loss = optimizer.step(closure).item()
        # prediction = func(train_x, param) 
        # train_r2 = calculate_r_squared(train_y, prediction)
        with torch.no_grad():
            eval_loss = 0
            for func, y in zip(funcs, train_y):
                eval_prediction = func(train_x, param)
                if loss_type == "huber":
                    eval_loss += torch.nn.functional.huber_loss(y, eval_prediction, delta=delta, reduction="sum").item()
                elif loss_type == "mse":
                    eval_loss += torch.nn.functional.mse_loss(y, eval_prediction, reduction="sum").item()

                #eval_loss += torch.nn.functional.huber_loss(eval_prediction, y, delta=delta).item() 
                # eval_loss = -calculate_r_squared(train_y, eval_prediction)
                # eval_loss = -eval_r2
        if eval_loss <= min_loss: # FIXME
            min_loss = eval_loss
            best_param = param.detach().clone()
            best_step = _
        # print(loss)
        if np.abs(min_loss - eval_loss) < eps:
            assert False
            break
    #print(f"min_loss: {min_loss}, best param: {best_param} , init: {init_param}")

    return min_loss, best_param, best_step


class MultiObjScalingLaw:
    def __init__(self, funcs):
        self.funcs = funcs
        self.params = None
        
    def fit(self, x, ys, init_params, max_step=20, eps=0, workers=-1, valid_split=0, delta=0.01, loss_type="huber"):
        if workers == -1:
            workers = mp.cpu_count()
        init_params = [torch.tensor(init_param, dtype=torch.float32) for init_param in init_params]
        minloss, optimal_param = 1e10, None
        _fit = partial(fit_multi_obj_scaling_laws, self.funcs, valid_split, x, ys, max_step, eps, delta, loss_type)
        print(f"workers: {workers}")
        if workers != 1:
            best_step = 0
            with mp.Pool(workers) as p:
                for loss, param, step in tqdm(p.imap_unordered(_fit, init_params, chunksize=2), total=len(init_params)):
                    if loss < minloss:
                        minloss = loss
                        optimal_param = param
                        best_step = step
        else:
            for init_param in tqdm(init_params):
                loss, param, step = _fit(init_param)
                # print(param)
                if loss < minloss:
                    minloss = loss
                    optimal_param = param 
        self.params = optimal_param.tolist()
        print(f"min loss: {minloss}")
        print(f"optimal_param: {optimal_param}")
        return self.params


