from flax import linen as nn

class L0ArmDense(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x, mode="normal"):
        def apply_mask(variables, mask):
            if not self.is_initializing():
                variables['params']['kernel'] *= mask
            return variables

        _MaskedDense = nn.map_variables(
            nn.Dense, mapped_collections="params", 
            trans_in_fn=apply_mask, mutable="params")

        
        rng = self.make_rng("l0")
        u = self.variable("l0", "u", nn.zeros, rng, _MaskedDense.variables['params']['kernel'])


        return _MaskedDense(self.features, name="masked_dense")(x)


    
