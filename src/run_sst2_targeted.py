"""Targeted SST-2 CNN/BiLSTM reproduction used for the COMP8240 project.

This is an architecture-level reproduction, not a byte-for-byte recreation of Wei & Zou (2019).
It uses 100 SST-2 training examples, 100 validation examples, trainable 300-d embeddings,
8 epochs and five seeds. Archived results used in the report are stored in results/.
"""

import random, re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

SEEDS=[1,7,19,42,99]
MAX_LEN=50
N_AUG=8

SYN={
    "good":["great","fine"],"great":["excellent","good"],"bad":["poor","awful"],
    "worst":["awful","poorest"],"movie":["film"],"film":["movie"],
    "funny":["humorous","amusing"],"beautiful":["lovely"],"interesting":["engaging"],
    "boring":["dull"],"slow":["sluggish"],"love":["adore"],"enjoyable":["pleasant"],
    "excellent":["great"],"awful":["terrible"],"terrible":["awful"],"smart":["clever"],
    "charming":["delightful"],"strong":["powerful"],"weak":["poor"],"story":["tale"],
    "performance":["acting"],"performances":["acting"],"director":["filmmaker"],
    "actors":["performers"],"actor":["performer"]
}
STOP=set("a an the and or but if then than is are was were be been being i my me you your we our it this that to of for in on at with from as by about before after still yet can could would should do did does have has had not no".split())

def toks(text):
    return re.findall(r"[A-Za-z']+|[.,!?;:$()-]", text.lower())

def untok(ws):
    return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(ws))

def sr(ws,r):
    ids=[i for i,w in enumerate(ws) if w in SYN and w not in STOP]
    if not ids: return ws[:]
    z=ws[:]; i=r.choice(ids); z[i]=r.choice(SYN[ws[i]]); return z

def ri(ws,r):
    cand=[w for w in ws if w in SYN and w not in STOP]
    if not cand: return ws[:]
    w=r.choice(cand); z=ws[:]; z.insert(r.randrange(len(z)+1),r.choice(SYN[w])); return z

def rs(ws,r):
    ids=[i for i,w in enumerate(ws) if re.fullmatch(r"[A-Za-z']+",w)]
    if len(ids)<2: return ws[:]
    z=ws[:]; a,b=r.sample(ids,2); z[a],z[b]=z[b],z[a]; return z

def rd(ws,r,p=.1):
    z=[w for w in ws if not re.fullmatch(r"[A-Za-z']+",w) or r.random()>p]
    return z if any(re.fullmatch(r"[A-Za-z']+",w) for w in z) else ws[:1]

def augment(text,seed,n=N_AUG):
    r=random.Random(seed); ws=toks(text); ops=[sr,ri,rs,rd]
    return [untok(ops[i%4](ws,r)) for i in range(n)]

raw=load_dataset("glue","sst2")
train=[(x["sentence"],int(x["label"])) for x in raw["train"].select(range(100))]
val=[(x["sentence"],int(x["label"])) for x in raw["validation"].select(range(100))]

words=[]
for t,_ in train:
    words += [w for w in toks(t) if re.fullmatch(r"[A-Za-z']+",w)]
vocab={"<pad>":0,"<unk>":1}
for w,_ in Counter(words).most_common():
    vocab[w]=len(vocab)

def enc(text):
    ids=[vocab.get(w,1) for w in toks(text) if re.fullmatch(r"[A-Za-z']+",w)][:MAX_LEN]
    return ids+[0]*(MAX_LEN-len(ids))

class TextDS(Dataset):
    def __init__(self,items):
        self.x=torch.tensor([enc(t) for t,_ in items],dtype=torch.long)
        self.y=torch.tensor([y for _,y in items],dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]

class CNN(nn.Module):
    def __init__(self,nv):
        super().__init__()
        self.emb=nn.Embedding(nv,300,padding_idx=0)
        self.conv=nn.Conv1d(300,128,5)
        self.fc1=nn.Linear(128,20); self.fc2=nn.Linear(20,2)
    def forward(self,x):
        x=self.emb(x).transpose(1,2)
        x=torch.relu(self.conv(x))
        x=torch.max(x,dim=2).values
        return self.fc2(torch.relu(self.fc1(x)))

class BiLSTM(nn.Module):
    def __init__(self,nv):
        super().__init__()
        self.emb=nn.Embedding(nv,300,padding_idx=0)
        self.l1=nn.LSTM(300,64,batch_first=True,bidirectional=True)
        self.l2=nn.LSTM(128,32,batch_first=True,bidirectional=True)
        self.drop=nn.Dropout(.5)
        self.fc1=nn.Linear(64,20); self.fc2=nn.Linear(20,2)
    def forward(self,x):
        x,_=self.l1(self.emb(x)); x=self.drop(x)
        _,(h,_)=self.l2(x)
        x=torch.cat([h[-2],h[-1]],1)
        return self.fc2(torch.relu(self.fc1(self.drop(x))))

def run(model_cls,use_eda,seed,epochs=8):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    items=list(train)
    if use_eda:
        extra=[]
        for j,(t,y) in enumerate(train):
            extra += [(a,y) for a in augment(t,seed*1000+j)]
        items += extra
    dl=DataLoader(TextDS(items),batch_size=32,shuffle=True,
                  generator=torch.Generator().manual_seed(seed))
    model=model_cls(len(vocab))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    lossfn=nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb,yb in dl:
            opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step()
    model.eval(); preds=[]; ys=[]
    with torch.no_grad():
        for xb,yb in DataLoader(TextDS(val),batch_size=64):
            preds += model(xb).argmax(1).tolist(); ys += yb.tolist()
    return sum(a==b for a,b in zip(preds,ys))/len(ys)

if __name__=="__main__":
    for name,cls in [("CNN",CNN),("BiLSTM",BiLSTM)]:
        for use_eda in [False,True]:
            vals=[]
            for seed in SEEDS:
                acc=run(cls,use_eda,seed); vals.append(acc)
                print(name,"EDA" if use_eda else "Baseline",seed,f"{acc:.3f}")
            print(name,"EDA" if use_eda else "Baseline","mean",f"{np.mean(vals):.3f}")
