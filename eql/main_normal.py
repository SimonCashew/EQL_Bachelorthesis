import jax
from jax import lax, random, numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import sympy as sy
from sympy.core.rules import Transform
import numpy as np
import optax
import scipy

import sys
sys.path.append("..")
sys.path.append("../../orient/")

from eql.eqlearner import EQLdiv
from eql.symbolic import get_symbolic_expr_div, get_symbolic_expr
from eql.np_utils import flatten, unflatten

import wandb
import hydra
from hydra import initialize, compose
import omegaconf
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    print("Starting run...")
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project="eval_normal_3", config=cfg_container)
    cfg_dict = dict(wandb.config)
    cfg = OmegaConf.create(cfg_dict)
    print("Backend:", jax.default_backend())
    print("Devices:", jax.devices())

    xdim = cfg.in_size
    width = cfg.width_size
    depth = cfg.depth
    key = random.PRNGKey(cfg.seed)
    init_key = random.PRNGKey(cfg.init_seed)
    lr_start = cfg.lr_start
    reg_factor = cfg.reg_factor

    batchsize = 10000
    l0_threshold = 1e-3
    noise_key = random.PRNGKey(3)
    noise = random.normal(noise_key, shape=(9000,)) * 0.01

    if (width == 12):
        funs = ['mul', 'mul', 'sin', 'cos', 'id', 'id']*2
    elif (width == 15):
        funs = ['mul', 'sin', 'cos', 'id', 'id']*3
    elif (width == 18):
        funs = ['mul', 'mul', 'cos', 'cos', 'sin', 'sin', 'id', 'id', 'id']*2
    e = EQLdiv(n_layers=depth, functions=funs, features=1)

    x = random.uniform(key, (9000, xdim), minval=-1.0, maxval=1.0)
    x_val = random.uniform(random.PRNGKey(1), (1000, xdim), minval=-1.0, maxval=1.0)
    x_raw = random.uniform(random.PRNGKey(2), (120, xdim), minval=-2.0, maxval=2.0)
    x_val_ex = x_raw[(jnp.any(x_raw < -1.0, axis=1) | jnp.any(x_raw > 1.0, axis=1))][:40]
    x_test = random.uniform(random.PRNGKey(3), (1000, xdim), minval=-1.0, maxval=1.0)
    x_raw = random.uniform(random.PRNGKey(4), (120, xdim), minval=-2.0, maxval=2.0)
    x_test_ex = x_raw[(jnp.any(x_raw < -1.0, axis=1) | jnp.any(x_raw > 1.0, axis=1))][:40]

    if (xdim == 2):
        y = np.sin(np.pi * x[:,0])/(x[:,1]**2 + 1) + noise
        y_val = np.sin(np.pi * x_val[:,0])/(x_val[:,1]**2 + 1)
        y_val_ex = np.sin(np.pi * x_val_ex[:,0])/(x_val_ex[:,1]**2 + 1)
        y_test = np.sin(np.pi * x_test[:,0])/(x_test[:,1]**2 + 1)
        y_test_ex = np.sin(np.pi * x_test_ex[:,0])/(x_test_ex[:,1]**2 + 1)
        max_out = 3
    elif (xdim == 3):
        y = 1./2.*(np.sin(np.pi*x[:,0]) + np.cos(2*x[:,1]*np.sin(np.pi*x[:,0]))-x[:,1]*x[:,2]) + noise
        y_val = 1./2.*(np.sin(np.pi*x_val[:,0]) + np.cos(2*x_val[:,1]*np.sin(np.pi*x_val[:,0]))-x_val[:,1]*x_val[:,2])
        y_val_ex = 1./2.*(np.sin(np.pi*x_val_ex[:,0]) + np.cos(2*x_val_ex[:,1]*np.sin(np.pi*x_val_ex[:,0]))-x_val_ex[:,1]*x_val_ex[:,2])
        y_test = 1./2.*(np.sin(np.pi*x_test[:,0]) + np.cos(2*x_test[:,1]*np.sin(np.pi*x_test[:,0]))-x_test[:,1]*x_test[:,2])
        y_test_ex = 1./2.*(np.sin(np.pi*x_test_ex[:,0]) + np.cos(2*x_test_ex[:,1]*np.sin(np.pi*x_test_ex[:,0]))-x_test_ex[:,1]*x_test_ex[:,2])
        max_out = 9
    elif (xdim == 4):
        y = 1./3. * ((1.+x[:,1])*np.sin(np.pi*x[:,0]) + x[:,1]*x[:,2]*x[:,3]) + noise
        y_val = 1./3. * ((1.+x_val[:,1])*np.sin(np.pi*x_val[:,0]) + x_val[:,1]*x_val[:,2]*x_val[:,3])
        y_val_ex = 1./3. * ((1.+x_val_ex[:,1])*np.sin(np.pi*x_val_ex[:,0]) + x_val_ex[:,1]*x_val_ex[:,2]*x_val_ex[:,3])
        y_test = 1./3. * ((1.+x_test[:,1])*np.sin(np.pi*x_test[:,0]) + x_test[:,1]*x_test[:,2]*x_test[:,3])
        y_test_ex = 1./3. * ((1.+x_test_ex[:,1])*np.sin(np.pi*x_test_ex[:,0]) + x_test_ex[:,1]*x_test_ex[:,2]*x_test_ex[:,3])
        max_out = 11

    params = e.init({'params':init_key}, x, 1.0)

    def mse_fn(params, threshold):
        pred, _ = e.apply(params, x, threshold)
        return jnp.mean((pred-y)**2)

    def mse_b_fn(params, threshold):
        pred, b = e.apply(params, x, threshold)
        return jnp.mean((pred-y)**2), b

    def get_mask_spec(thresh, params):
        flat, spec = tree_flatten(params)
        mask = [jnp.abs(f) > thresh for f in flat]
        return mask, spec

    def apply_mask(mask, spec, params):
        flat, _ = tree_flatten(params)
        masked_params = tree_unflatten(spec, [f*m for f,m in zip(flat, mask)])
        return masked_params
    

    def l1_fn(params):
        return sum(
            jnp.abs(w).mean() for w in jax.tree.leaves(params["params"])
        )

    def reg_fn(threshold, b):
        return (jnp.maximum(0, threshold - b)).sum()

    def penalty_fn(y, B=max_out):
        return jnp.sum(jnp.maximum(y-B, 0)+jnp.maximum(-y-B, 0))

    def get_loss(lamba):
        def loss_fn(params, threshold):
            mse, b = mse_b_fn(params, threshold)
            return mse  + lamba * l1_fn(params) + reg_fn(threshold, b)
        return loss_fn

    def get_loss_pen():
        def loss_fn(params, threshold): 
            xr = random.uniform(random.PRNGKey(i), (batchsize, xdim), minval=-2.0, maxval=2.0)
            y, b = e.apply(params, xr, threshold)
            return penalty_fn(y) + reg_fn(threshold, b)
        return loss_fn

    def get_loss_grad(lamba = 1e-3, is_penalty=False):
        if is_penalty:
            loss = get_loss_pen()
        else:
            loss = get_loss(lamba)
        return jax.jit(jax.value_and_grad(loss))
        
    tx = optax.adam(learning_rate=lr_start)
    opt_state = tx.init(params)

    loss_grad_pen = get_loss_grad(is_penalty=True)
    loss_grad_1 = get_loss_grad(0)
    loss_grad_2 = get_loss_grad(reg_factor)

    def do_step(loss_grad, params, theta, opt_state):
        loss_val, grad = loss_grad(params, theta)
        updates, opt_state = tx.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss_val
    
    T1 = 5000
    for i in range(19000):
        theta = 1./jnp.sqrt(i/1 + 1)
        if i < T1:
            lg = loss_grad_1
        elif i >= T1:
            lg = loss_grad_2
        params, opt_state, loss_val = do_step(lg, params, theta, opt_state)
        if i % 50 == 0 and i > 0:
            run.log({"loss": loss_val})
            params, opt_state, loss_val = do_step(loss_grad_pen, params, theta, opt_state)
        
    thr = l0_threshold
    T = 19000
    lg = loss_grad_1
    for i in range(1000):
        theta = 1./jnp.sqrt(T/1 + 1)
        params, opt_state, loss_val = do_step(lg, params, theta, opt_state)
        mask, spec = get_mask_spec(thr, params)
        params = apply_mask(mask, spec, params)
        T +=1
        if i % 50 == 0:
            run.log({"loss": loss_val})
            params, opt_state, loss_val = do_step(loss_grad_pen, params, theta, opt_state)

    def mse_val_fn(params, threshold):
        pred, _ = e.apply(params, x_val, threshold)
        return jnp.mean((pred - y_val) ** 2)

    val_loss = mse_val_fn(params, 1e-4)

    table = wandb.Table(columns=["Sweep", "Validation Loss"])
    table.add_data(wandb.run.name, val_loss)
    run.log({"Validation Sweep Table": table})
    run.log({"Validation Loss": val_loss})

    def mse_val_ex_fn(params, threshold):
        pred, _ = e.apply(params, x_val_ex, threshold)
        return jnp.mean((pred - y_val_ex) ** 2)

    val_ex_loss = mse_val_ex_fn(params, 1e-4)

    table3 = wandb.Table(columns=["Sweep", "Extrapolation Validation Loss"])
    table3.add_data(wandb.run.name, val_ex_loss)
    run.log({"Extrapolation Validation Sweep Table": table3})
    run.log({"Extrapolation Validation Loss": val_ex_loss})

    def log_zeros(params):
        flattened_params, _ = tree_flatten(params)

        total_zeros = sum(jnp.sum(param == 0).item() for param in flattened_params if isinstance(param, jnp.ndarray))
        total_params = sum(param.size for param in flattened_params if isinstance(param, jnp.ndarray))
        table2 = wandb.Table(columns=["Sweep", "Parameters", "Total Parameters"])
        table2.add_data(wandb.run.name, total_params - total_zeros, total_params)
        run.log({"Parameter Sweep Table": table2})
        run.log({"Parameter": total_params - total_zeros})

    log_zeros(params)

    def mse_test_fn(params, threshold):
        pred, _ = e.apply(params, x_test, threshold)
        return jnp.mean((pred - y_test) ** 2)

    test_loss = mse_test_fn(params, 1e-4)

    def mse_test_ex_fn(params, threshold):
        pred, _ = e.apply(params, x_test_ex, threshold)
        return jnp.mean((pred - y_test_ex) ** 2)

    test_ex_loss = mse_test_ex_fn(params, 1e-4)
    run.log({"Test Loss": test_loss})
    run.log({"Extrapolation Test Loss": test_ex_loss})

    wandb.finish()


    symb = get_symbolic_expr_div(params, funs)[0]

    spec, fparam = flatten(params)
    full_shape = fparam.shape
    mask = jnp.abs(fparam) > 0.01
    idxs = jnp.arange(fparam.shape[0])[mask]
    count = sum(mask).item()

    def red_loss_grad_fn(red_param):
        full_param = jnp.zeros(full_shape).at[idxs].set(red_param)
        full_param = unflatten(spec, full_param)

        loss, grad = loss_grad_1(full_param, 1e-4)
        _, grad = flatten(grad)
        return loss, np.array(grad)[idxs,]
    
    x0, f, info = scipy.optimize.fmin_l_bfgs_b(
        red_loss_grad_fn,
        x0 = np.array(fparam[mask]),
        factr=1.,
        m=500,
        pgtol=1e-13,
        maxls=100)
    
    final_param = unflatten(spec, jnp.zeros(full_shape).at[idxs].set(x0))

    symb = get_symbolic_expr_div(final_param, funs)[0]

    def clean_expr(expr):
        def prune(expr, thr=1e-5):
            return expr.replace(lambda x: x.is_Number and abs(x) < thr, lambda x: 0)
        
        def rounding(expr, dig=3):
            return expr.xreplace(Transform(lambda x: x.round(dig), lambda x: x.is_Number))
        
        expr = prune(expr)
        expr = rounding(expr)
        expr = prune(sy.expand(expr), 1e-3)
        print(expr)
        print(sy.simplify(expr))
    
    clean_expr(symb)    

if __name__=='__main__':
    train()
