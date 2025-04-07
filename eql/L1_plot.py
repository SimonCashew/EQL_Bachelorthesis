import os
import numpy as np
from scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as matcolors
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set_style("ticks")

from einops import rearrange
p_opt = jnp.array([-1., -1.])

def loss_datafit(x, y):
    return jnp.sum((x - y)**2)

def loss_L1reg(p):
    return jnp.sum(jnp.abs(p))

@jax.value_and_grad
def loss_sum(p, y):
    return loss_datafit(p, y) + 1.0 * loss_L1reg(p)

n = 50
x = jnp.linspace(-2, 1, n)
y = jnp.linspace(-2, 1, n)
xv, yv = np.meshgrid(x, y)
p = jnp.c_[xv.flatten(), yv.flatten()]

losses_datafit, grads_datafit = jax.vmap(jax.value_and_grad(loss_datafit), in_axes=(0, None))(p, p_opt)
losses_L1reg, grads_L1reg = jax.vmap(jax.value_and_grad(loss_L1reg))(p)
def orth_proj(u, v):
    dot_product = jnp.dot(u, v)
    v_norm_squared = jnp.linalg.norm(v)**2
    proj = (dot_product/v_norm_squared) * v
    return u - proj
grads_OPRreg = jax.vmap(orth_proj)(grads_L1reg, grads_datafit)
losses_sum, _ = jax.vmap(loss_sum, in_axes=(0, None))(p, p_opt)
grads_sum = grads_datafit + grads_OPRreg

titles = ["Datafit", "OPR", "Sum"]
losses = [losses_datafit, losses_L1reg, losses_sum]
grads = [grads_datafit, grads_OPRreg, grads_sum]

fig, ax = plt.subplots(2, 3, sharex=True, squeeze=False)
for i, (l, g) in enumerate(zip(losses, grads)):
    CS = ax[0,i].contourf(xv, yv, l.reshape(xv.shape), 10, alpha=0.5, cmap="Blues")
    ax[0,i].contour(CS, levels=CS.levels[::2], colors='k', alpha=0.5)

    grads_scaled = jax.vmap(lambda gi: gi / jnp.linalg.norm(gi))(g)
    length_grad_sum = jnp.linalg.norm(g, axis=1)
    take_every = 4
    ax[1,i].quiver(p[::take_every, 0], 
                   p[::take_every, 1], 
                   -grads_scaled[::take_every, 0], 
                   -grads_scaled[::take_every, 1],
                   length_grad_sum[::take_every],
                   units="width",
                   pivot="mid",
                   scale=30.0,
                   headwidth=3,
                   width=0.01,
                   cmap="coolwarm"
                   )
    
    ax[0,i].set_title(titles[i], fontsize=20)
sns.despine()
plt.tight_layout()
plt.show()
