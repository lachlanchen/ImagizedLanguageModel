"""Exact suffix counts over physical cell identities, with a smoothed unigram fallback."""
import numpy as np

DEPTH=6
SMOOTH=2.0


def history_windows(stream,records):
    hist=np.zeros((len(stream),DEPTH),dtype=np.uint16)
    for *_,start,end in records:
        for lag in range(1,min(DEPTH+1,end-start)):
            hist[start+lag:end,lag-1]=stream[start:end-lag]
    return hist


def find_keys(keys,query):
    idx=keys.searchsorted(query)
    if not len(keys):
        return idx,np.zeros(query.shape,bool)
    return idx,keys[idx.clip(max=len(keys)-1)]==query


class PrefixCounts:
    def __init__(self,arrays):
        self.arrays=arrays
        for k,v in arrays.items():
            setattr(self,k,v)

    @classmethod
    def build(cls,history,truth,vocab):
        prior=np.bincount(truth,minlength=vocab+1).astype(np.float64)+0.5
        prior[0]=0
        prior=(prior/prior.sum()).astype(np.float32)
        active=np.arange(len(truth))
        parent=np.zeros(len(truth),np.int64)
        starts=[0]
        keys_by_depth=[]
        edges=[]
        counts=[]
        for d in range(DEPTH):
            active=active[history[active,d]>0]
            keys,inv,n=np.unique((parent[active]<<16)+history[active,d],return_inverse=True,return_counts=True)
            admitted=n>=2
            keep=admitted[inv]
            active=active[keep]
            keys_by_depth.append(keys[admitted])
            parent[active]=starts[-1]+np.cumsum(admitted)[inv[keep]]
            starts.append(starts[-1]+len(keys_by_depth[-1]))
            edge,count=np.unique((parent[active]<<16)+truth[active],return_counts=True)
            edges.append(edge)
            counts.append(count.astype(np.int32))
        edge=np.concatenate(edges)
        freq=np.concatenate(counts)
        node=edge>>16
        total=np.bincount(node,weights=freq,minlength=starts[-1]+1).astype(np.int32)
        peak=np.zeros_like(total)
        np.maximum.at(peak,node,freq)
        ptr=np.r_[0,np.cumsum(np.bincount(node,minlength=len(total)))]
        return cls(dict(prior=prior,keys=np.concatenate(keys_by_depth),starts=np.array(starts),
                        edges=edge,freq=freq,ptr=ptr,total=total,peak=peak))

    def lookup(self,history):
        node=np.zeros(len(history),np.int64)
        depth=np.zeros(len(history),np.int8)
        live=np.arange(len(history))
        for d,(lo,hi) in enumerate(zip(self.starts[:-1],self.starts[1:])):
            if not len(live):
                break
            ix,ok=find_keys(self.keys[lo:hi],(node[live]<<16)+history[live,d])
            live=live[ok]
            node[live]=lo+1+ix[ok]
            depth[live]=d+1
        n=self.total[node]
        concentration=np.where(node==0,self.prior.max(),self.peak[node]/np.maximum(n,1))
        features=np.stack([depth/DEPTH,np.log1p(n)/8,concentration],axis=-1).astype(np.float32)
        return node,features

    def target_prob(self,node,truth):
        idx,ok=find_keys(self.edges,(node<<16)+truth)
        hit=np.zeros(len(node),np.float32)
        hit[ok]=self.freq[idx[ok]]
        prior=np.zeros_like(hit)
        known=(truth>=0)&(truth<len(self.prior))
        prior[known]=self.prior[truth[known]]
        return ((hit+SMOOTH*prior)/(self.total[node]+SMOOTH)).astype(np.float32)

    def dense(self,node):
        mass=np.broadcast_to(self.prior,(len(node),len(self.prior))).copy()*SMOOTH
        lengths=self.ptr[node+1]-self.ptr[node]
        rows=np.repeat(np.arange(len(node)),lengths)
        if len(rows):
            begin=np.cumsum(lengths)-lengths
            entry=np.arange(len(rows))-begin[rows]+self.ptr[node[rows]]
            mass[rows,self.edges[entry]&65535]+=self.freq[entry]
        mass/=self.total[node,None]+SMOOTH
        return mass
