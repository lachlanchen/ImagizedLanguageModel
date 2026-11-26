"""Conditional probabilities of a glyph stream under all segmentations into learned image strips."""
import numpy as np


class StripInventory:
    def __init__(self,strips,output_ids):
        self.index={():0}
        self.next=[{}]
        self.terminal=[0]
        self.paths=[]
        self.width=max(map(len,strips))
        self.size=len(output_ids)
        self.global_to_local={v:i for i,v in enumerate(output_ids)}
        for k,strip in enumerate(strips):
            path=[]
            parent=0
            for j,c in enumerate(strip):
                prefix=tuple(strip[:j+1])
                if prefix not in self.index:
                    self.index[prefix]=len(self.next)
                    self.next.append({})
                    self.terminal.append(0)
                child=self.index[prefix]
                self.next[parent][self.global_to_local[c]]=child
                parent=child
                path.append(child)
            self.paths.append(path)
            if k:
                self.terminal[parent]=k
        self.terminal=np.array(self.terminal,np.int32)
        self.child=[(np.fromiter(d,np.int32),np.fromiter(d.values(),np.int32)) for d in self.next]
        self.parent_nodes=np.array([n for path in self.paths for n in path],np.int32)
        self.parent_labels=np.repeat(np.arange(len(strips)),list(map(len,self.paths)))

    def source_strips(self,cells):
        # An unrepresented input bitmap contributes no lexical match. The neural path retains it.
        result=[0]
        i=0
        while i<len(cells):
            for k in range(min(self.width,len(cells)-i),0,-1):
                node=self.index.get(tuple(cells[i:i+k]))
                if node is not None and self.terminal[node]:
                    result.append(self.terminal[node])
                    break
            else:
                k=1
            i+=k
        return result


class StripPrefix:
    def __init__(self,inventory,law):
        self.vocab=inventory
        self.law=np.asarray(law,np.float64)
        self.mass=np.bincount(inventory.parent_nodes,weights=self.law[inventory.parent_labels],minlength=len(inventory.next))
        self.log_boundaries=[0.]
        self.history=[]
        self.boundary=None

    def probabilities(self):
        vocab=self.vocab
        branches=[(0,self.log_boundaries[-1])]
        for k in range(1,min(len(self.history)+1,vocab.width)):
            n=vocab.index.get(tuple(self.history[-k:]))
            if n is not None and vocab.next[n]:
                branches.append((n,self.log_boundaries[-k-1]))
        self.base=max(weight for _,weight in branches)
        mass=np.zeros(vocab.size)
        self.boundary=np.zeros(vocab.size)
        for n,log_weight in branches:
            w=np.exp(log_weight-self.base)
            chars,child=vocab.child[n]
            mass[chars]+=w*self.mass[child]
            self.boundary[chars]+=w*self.law[vocab.terminal[child]]
        return mass/mass.sum()

    def advance(self,local_id,physical_id):
        if self.boundary is None or self.boundary[local_id]<=0:
            raise ValueError("No positive strip segmentation for next bitmap")
        self.log_boundaries.append(self.base+np.log(self.boundary[local_id]))
        self.history.append(physical_id)
        self.boundary=None


def expand_table(ids,mass,tail,prior):
    # Partition each row into the stored heads and the prior restricted to its remaining strips.
    leftover=np.maximum(1-prior[ids].astype(np.float64).sum(1),1e-12)
    out=(tail/leftover)[:,None].astype(np.float32)*prior[None]
    np.put_along_axis(out,ids.astype(np.int32),mass,axis=1)
    out+=1e-8*prior
    out/=out.sum(1,keepdims=True)
    return out
