from pathlib import Path
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "customer_service_train.csv")
test = pd.read_csv(DATA / "customer_service_test.csv")

SYN = {
    "parcel":["package","shipment"], "package":["parcel","shipment"], "shipment":["parcel","package"],
    "order":["purchase"], "purchase":["order"], "delivery":["shipment"], "delivered":["arrived"],
    "arrived":["delivered","came"], "receive":["get"], "late":["delayed"], "delayed":["late"],
    "refund":["reimbursement"], "money":["payment"], "payment":["charge"], "charged":["billed"],
    "item":["product"], "product":["item"], "broken":["damaged"], "damaged":["broken"],
    "address":["location"], "shipping":["delivery"], "change":["update"], "update":["change"],
    "cancel":["stop"], "status":["progress"], "processing":["being prepared"], "dispatched":["shipped"],
    "shipped":["dispatched"], "warehouse":["depot"], "wrong":["incorrect"], "incorrect":["wrong"],
    "missing":["not received"], "confirm":["check"], "check":["confirm"], "current":["present"],
    "ready":["prepared"], "need":["want"], "want":["need"], "tell":["inform"]
}
STOP=set("a an the and or but if then than is are was were be been being i my me you your we our it this that to of for in on at with from as by about before after still yet can could would should please do did does have has had not no".split())

def tok(t): return re.findall(r"[A-Za-z']+|[.,!?;]", t.lower())
def untok(ws):
    s=" ".join(ws)
    s=re.sub(r"\s+([.,!?;])", r"\1", s)
    return s[:1].upper()+s[1:] if s else s

def sr(ws,n,r):
    ids=[i for i,w in enumerate(ws) if w.isalpha() and w not in STOP and w in SYN]
    r.shuffle(ids); z=ws[:]
    for i in ids[:n]: z[i]=r.choice(SYN[ws[i]])
    return z

def ri(ws,n,r):
    z=ws[:]
    for _ in range(n):
        c=[w for w in z if w.isalpha() and w not in STOP and w in SYN]
        if not c: break
        w=r.choice(c); pos=r.randint(0,len(z)); z[pos:pos]=r.choice(SYN[w]).split()
    return z

def rs(ws,n,r):
    z=ws[:]; ids=[i for i,w in enumerate(z) if w.isalpha()]
    for _ in range(n):
        if len(ids)<2: break
        a,b=r.sample(ids,2); z[a],z[b]=z[b],z[a]
    return z

def rd(ws,p,r):
    z=[w for w in ws if (not w.isalpha()) or r.random()>p]
    return z if any(w.isalpha() for w in z) else [r.choice([w for w in ws if w.isalpha()])]

def augment(t,r,num_aug=8,alpha=.1):
    ws=tok(t); n=max(1,int(alpha*max(1,sum(w.isalpha() for w in ws))))
    ops=[lambda:sr(ws,n,r),lambda:ri(ws,n,r),lambda:rs(ws,n,r),lambda:rd(ws,alpha,r)]
    return [untok(ops[i%4]()) for i in range(num_aug)]

def evaluate(num_aug, seed):
    x=list(train.text); y=list(train.label); r=random.Random(seed)
    if num_aug:
        for text,label in zip(train.text,train.label):
            for v in augment(text,r,num_aug=num_aug):
                x.append(v); y.append(label)
    model=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
        ("clf",LogisticRegression(max_iter=2000,C=4.0,random_state=seed))
    ])
    model.fit(x,y)
    pred=model.predict(test.text)
    return accuracy_score(test.label,pred), f1_score(test.label,pred,average="macro")

seeds=[1,7,19,42,99]
rows=[]
for seed in seeds:
    bacc,bf1=evaluate(0,seed)
    eacc,ef1=evaluate(8,seed)
    rows.append({"seed":seed,"baseline_accuracy":bacc,"baseline_macro_f1":bf1,
                 "eda_accuracy":eacc,"eda_macro_f1":ef1,
                 "accuracy_change_pp":(eacc-bacc)*100})
res=pd.DataFrame(rows)
res.to_csv(RESULTS/"final_experiment_results.csv",index=False)
print(res.to_string(index=False))
print("\nMeans:")
print(res.mean(numeric_only=True).to_string())
