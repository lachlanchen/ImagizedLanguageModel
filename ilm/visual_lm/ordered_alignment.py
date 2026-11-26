"""Exact glyph-prefix probabilities in a source-position hidden Markov model."""
import numpy as np


def movement(size, weights):
    weights=np.asarray(weights,dtype=np.float64)
    if size<1 or weights.shape!=(5,) or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError("Expected a nonempty source and five movement probabilities")
    weights=weights/weights.sum()
    kernel=np.full((size,size),weights[0]/size)
    rows=np.arange(size)
    for offset,p in zip((-1,0,1,2),weights[1:],strict=True):
        kernel[rows,(rows+offset).clip(0,size-1)]+=p
    return kernel


def emission_laws(table, ids):
    """The uniform position mixture is the original null-inclusive lexical model."""
    n=len(ids)
    if not n:
        return table[[0]].astype(np.float64)
    return (n*table[ids].astype(np.float64)+table[0])/(n+1)


def strip_log_likelihood(emission, transition, target):
    state=np.full(len(emission),1/len(emission))
    log_mass=0.
    for k in target:
        attached=state*emission[:,k]
        scale=attached.sum()
        log_mass+=np.log(scale)
        state=(attached/scale)@transition
    return log_mass


class AlignedPrefix:
    """Filter alignment at each viable strip boundary, including unfinished strips."""
    def __init__(self,inventory,emission,kernel):
        self.vocab=inventory
        self.emit=emission
        self.kernel=kernel
        self.history=[]
        # Each boundary stores its log mass and the conditional law of the next source position.
        self.frontier=[(0.,np.full(len(emission),1/len(emission)))]
        self.subtree=np.stack([np.bincount(inventory.parent_nodes,weights=row[inventory.parent_labels],
                                          minlength=len(inventory.next)) for row in emission])
        self.branches=None

    def probabilities(self):
        v=self.vocab
        nodes=[(0,self.frontier[-1])]
        for k in range(1,min(len(self.history)+1,v.width)):
            node=v.index.get(tuple(self.history[-k:]))
            if node is not None and v.next[node]:
                nodes.append((node,self.frontier[-k-1]))
        self.base=max(front[0] for node,front in nodes)
        self.branches=[]
        mass=np.zeros(v.size)
        for node,(log_mass,state) in nodes:
            local,children=v.child[node]
            scaled=state*np.exp(log_mass-self.base)
            mass[local]+=self.subtree[:,children].T@scaled
            self.branches.append((node,scaled))
        return mass/mass.sum()

    def advance(self,local_id,physical_id):
        if self.branches is None:
            raise ValueError("Score the prefix before advancing it")
        post=np.zeros(len(self.emit))
        for node,scaled in self.branches:
            child=self.vocab.next[node].get(local_id)
            if child is not None and self.vocab.terminal[child]:
                post+=self.emit[:,self.vocab.terminal[child]]*scaled
        scale=post.sum()
        if not scale>0:
            raise ValueError("No compatible completed strip at this boundary")
        self.frontier.append((self.base+np.log(scale),(post/scale)@self.kernel))
        self.history.append(physical_id)
        self.branches=None
