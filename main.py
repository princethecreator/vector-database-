import math, heapq, random, time, json, requests, uuid
from collections import defaultdict
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, os

def cosine_distance(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    ma = math.sqrt(sum(x*x for x in a))
    mb = math.sqrt(sum(x*x for x in b))
    return 1.0 - dot/(ma*mb) if ma and mb else 1.0

def euclidean_distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def manhattan_distance(a, b):
    return sum(abs(x-y) for x,y in zip(a,b))

def get_dist(metric):
    return {"cosine":cosine_distance,"euclidean":euclidean_distance,"manhattan":manhattan_distance}.get(metric,cosine_distance)

class BruteForce:
    def __init__(self): self.items=[]
    def insert(self,item): self.items.append(item)
    def delete(self,id): self.items=[x for x in self.items if x["id"]!=id]
    def search(self,q,k,metric="cosine"):
        fn=get_dist(metric)
        s=sorted([(fn(q,i["vector"]),i) for i in self.items],key=lambda x:x[0])
        return [{"item":i,"distance":d} for d,i in s[:k]]

class KDNode:
    __slots__=["item","left","right","dim"]
    def __init__(self,item,dim): self.item=item;self.left=None;self.right=None;self.dim=dim

class KDTree:
    def __init__(self): self.root=None;self.dims=0
    def _build(self,items,depth):
        if not items: return None
        dim=depth%max(self.dims,1)
        items.sort(key=lambda x:x["vector"][dim])
        mid=len(items)//2
        node=KDNode(items[mid],dim)
        node.left=self._build(items[:mid],depth+1)
        node.right=self._build(items[mid+1:],depth+1)
        return node
    def build(self,items):
        if items: self.dims=len(items[0]["vector"])
        self.root=self._build(items,0)
    def insert(self,item):
        if not self.dims: self.dims=len(item["vector"])
        self.root=self._ins(self.root,item,0)
    def _ins(self,node,item,depth):
        if node is None: return KDNode(item,depth%self.dims)
        if item["vector"][depth%self.dims]<node.item["vector"][depth%self.dims]: node.left=self._ins(node.left,item,depth+1)
        else: node.right=self._ins(node.right,item,depth+1)
        return node
    def delete(self,id):
        all_items=[];self._col(self.root,all_items)
        all_items=[x for x in all_items if x["id"]!=id]
        self.root=self._build(all_items,0)
    def _col(self,node,out):
        if node is None: return
        out.append(node.item);self._col(node.left,out);self._col(node.right,out)
    def search(self,q,k,metric="cosine"):
        fn=get_dist(metric);best=[]
        self._srch(self.root,q,k,fn,best)
        best.sort(key=lambda x:x[0])
        return [{"item":i,"distance":d} for d,i in best]
    def _srch(self,node,q,k,fn,best):
        if node is None: return
        d=fn(q,node.item["vector"])
        if len(best)<k: heapq.heappush(best,(-d,node.item))
        elif d<-best[0][0]: heapq.heapreplace(best,(-d,node.item))
        diff=q[node.dim]-node.item["vector"][node.dim]
        near,far=(node.left,node.right) if diff<0 else (node.right,node.left)
        self._srch(near,q,k,fn,best)
        if len(best)<k or abs(diff)<-best[0][0]: self._srch(far,q,k,fn,best)

class HNSWNode:
    __slots__=["item","max_layer","neighbors"]
    def __init__(self,item,max_layer,M):
        self.item=item;self.max_layer=max_layer
        self.neighbors=[[] for _ in range(max_layer+1)]

class HNSW:
    def __init__(self,M=16,ef_c=200,ef_s=50):
        self.M=M;self.M0=M*2;self.ef_c=ef_c;self.ef_s=ef_s
        self.ml=1.0/math.log(M) if M>1 else 1.0
        self.nodes={};self.ep=None;self.max_layer=0
        self.dist_fn=cosine_distance
    def set_metric(self,m): self.dist_fn=get_dist(m)
    def _rl(self): return int(-math.log(random.random())*self.ml)
    def _dq(self,q,nid): return self.dist_fn(q,self.nodes[nid].item["vector"])
    def _sl(self,q,entry,ef,layer):
        ed=self._dq(q,entry)
        cands=[(ed,entry)];found=[(-ed,entry)];visited={entry}
        while cands:
            cd,cid=heapq.heappop(cands)
            if cd>-found[0][0] and len(found)>=ef: break
            for nb in self.nodes[cid].neighbors[layer]:
                if nb not in visited and nb in self.nodes:
                    visited.add(nb);nd=self._dq(q,nb)
                    if len(found)<ef or nd<-found[0][0]:
                        heapq.heappush(cands,(nd,nb));heapq.heappush(found,(-nd,nb))
                        if len(found)>ef: heapq.heappop(found)
        return [(-d,nid) for d,nid in found]
    def _sel(self,cands,M):
        cands.sort();return [nid for _,nid in cands[:M]]
    def insert(self,item):
        nid=item["id"];nl=self._rl()
        node=HNSWNode(item,nl,self.M);self.nodes[nid]=node
        if self.ep is None: self.ep=nid;self.max_layer=nl;return
        eid=self.ep;q=item["vector"]
        for layer in range(self.max_layer,nl,-1):
            c=self._sl(q,eid,1,layer);eid=min(c,key=lambda x:x[0])[1]
        for layer in range(min(nl,self.max_layer),-1,-1):
            c=self._sl(q,eid,self.ef_c,layer)
            Mm=self.M0 if layer==0 else self.M
            nbs=self._sel(c,Mm);node.neighbors[layer]=nbs
            for nb in nbs:
                if nb not in self.nodes: continue
                nn=self.nodes[nb]
                if layer<=nn.max_layer:
                    nn.neighbors[layer].append(nid)
                    if len(nn.neighbors[layer])>Mm:
                        nq=nn.item["vector"]
                        sc=[(self.dist_fn(nq,self.nodes[x].item["vector"]),x) for x in nn.neighbors[layer] if x in self.nodes]
                        nn.neighbors[layer]=self._sel(sc,Mm)
            if c: eid=min(c,key=lambda x:x[0])[1]
        if nl>self.max_layer: self.max_layer=nl;self.ep=nid
    def delete(self,id):
        if id not in self.nodes: return
        node=self.nodes[id]
        for layer in range(node.max_layer+1):
            for nb in node.neighbors[layer]:
                if nb in self.nodes:
                    nn=self.nodes[nb]
                    if layer<=nn.max_layer: nn.neighbors[layer]=[x for x in nn.neighbors[layer] if x!=id]
        del self.nodes[id]
        if self.ep==id: self.ep=next(iter(self.nodes),None);self.max_layer=max((n.max_layer for n in self.nodes.values()),default=0)
    def search(self,q,k,metric="cosine"):
        self.set_metric(metric)
        if not self.ep or self.ep not in self.nodes: return []
        eid=self.ep
        for layer in range(self.max_layer,0,-1):
            c=self._sl(q,eid,1,layer);eid=min(c,key=lambda x:x[0])[1]
        c=self._sl(q,eid,max(self.ef_s,k),0);c.sort()
        return [{"item":self.nodes[nid].item,"distance":d} for d,nid in c[:k] if nid in self.nodes]
    def info(self):
        lc=defaultdict(int);te=0
        for n in self.nodes.values():
            lc[n.max_layer]+=1
            for nb in n.neighbors: te+=len(nb)
        return {"total_nodes":len(self.nodes),"max_layer":self.max_layer,"M":self.M,"total_edges":te,"layer_distribution":dict(sorted(lc.items()))}

class VectorDB:
    def __init__(self): self.items=[];self.hnsw=HNSW();self.kd=KDTree();self.bf=BruteForce()
    def insert(self,item):
        self.items.append(item);self.hnsw.insert(item);self.kd.insert(item);self.bf.insert(item)
    def delete(self,id):
        self.items=[x for x in self.items if x["id"]!=id]
        self.hnsw.delete(id);self.kd.delete(id);self.bf.delete(id)
    def search(self,q,k,metric="cosine",algo="hnsw"):
        t=time.perf_counter()
        if algo=="hnsw": r=self.hnsw.search(q,k,metric)
        elif algo=="kdtree": r=self.kd.search(q,k,metric)
        else: r=self.bf.search(q,k,metric)
        return {"results":r,"time_us":(time.perf_counter()-t)*1e6,"algo":algo}
    def benchmark(self,q,k,metric="cosine"):
        return {a:self.search(q,k,metric,a) for a in ["hnsw","kdtree","brute"]}

OLLAMA_BASE="http://localhost:11434"
EMBED_MODEL="nomic-embed-text"
GEN_MODEL="llama3.2"

def ollama_embed(text):
    try:
        r=requests.post(f"{OLLAMA_BASE}/api/embeddings",json={"model":EMBED_MODEL,"prompt":text},timeout=60)
        return r.json()["embedding"]
    except: return None

def ollama_generate(prompt):
    try:
        with requests.post(f"{OLLAMA_BASE}/api/generate",json={"model":GEN_MODEL,"prompt":prompt,"stream":True},stream=True,timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    chunk=json.loads(line);yield chunk.get("response","")
                    if chunk.get("done"): break
    except Exception as e: yield f"\n[Error:{e}]"

def ollama_status():
    try:
        r=requests.get(f"{OLLAMA_BASE}/api/tags",timeout=5)
        models=[m["name"] for m in r.json().get("models",[])]
        return {"online":True,"models":models,"embed_model":EMBED_MODEL,"gen_model":GEN_MODEL}
    except: return {"online":False,"models":[],"embed_model":EMBED_MODEL,"gen_model":GEN_MODEL}

class DocumentDB:
    def __init__(self): self.hnsw=HNSW(M=16,ef_c=200);self.chunks={}
    def insert(self,title,text):
        words=text.split();cs,ol=250,50;chunks=[]
        i=0
        while i<len(words): chunks.append(" ".join(words[i:i+cs]));i+=cs-ol
        ids=[]
        for idx,chunk in enumerate(chunks):
            vec=ollama_embed(chunk)
            if vec is None: continue
            cid=str(uuid.uuid4())
            item={"id":cid,"vector":vec,"title":title,"text":chunk,"chunk_index":idx}
            self.hnsw.insert(item);self.chunks[cid]=item;ids.append(cid)
        return ids
    def delete(self,id): self.hnsw.delete(id);self.chunks.pop(id,None)
    def search(self,q,k=3): return self.hnsw.search(q,k,"cosine")
    def list(self): return [{"id":c["id"],"title":c["title"],"chunk_index":c["chunk_index"],"preview":c["text"][:120]+"..."} for c in self.chunks.values()]

DEMO=[
    {"id":"v1","label":"binary tree","category":"CS","vector":[0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v2","label":"hash table","category":"CS","vector":[0.85,0.75,0.65,0.55,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v3","label":"linked list","category":"CS","vector":[0.8,0.7,0.6,0.5,0.15,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v4","label":"graph algorithm","category":"CS","vector":[0.75,0.85,0.55,0.65,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v5","label":"dynamic programming","category":"CS","vector":[0.7,0.9,0.6,0.7,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v6","label":"calculus","category":"Math","vector":[0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v7","label":"linear algebra","category":"Math","vector":[0.1,0.1,0.1,0.1,0.85,0.75,0.65,0.55,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v8","label":"probability","category":"Math","vector":[0.1,0.1,0.1,0.1,0.8,0.7,0.6,0.5,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v9","label":"differential equations","category":"Math","vector":[0.1,0.1,0.1,0.1,0.75,0.85,0.55,0.65,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v10","label":"number theory","category":"Math","vector":[0.1,0.1,0.1,0.1,0.7,0.9,0.6,0.7,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id":"v11","label":"sushi","category":"Food","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1]},
    {"id":"v12","label":"pasta","category":"Food","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.85,0.75,0.65,0.55,0.1,0.1,0.1,0.1]},
    {"id":"v13","label":"tacos","category":"Food","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.7,0.6,0.5,0.1,0.1,0.1,0.1]},
    {"id":"v14","label":"ramen","category":"Food","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.75,0.85,0.55,0.65,0.1,0.1,0.1,0.1]},
    {"id":"v15","label":"pizza","category":"Food","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.7,0.9,0.6,0.7,0.1,0.1,0.1,0.1]},
    {"id":"v16","label":"basketball","category":"Sports","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6]},
    {"id":"v17","label":"soccer","category":"Sports","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.85,0.75,0.65,0.55]},
    {"id":"v18","label":"tennis","category":"Sports","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.7,0.6,0.5]},
    {"id":"v19","label":"swimming","category":"Sports","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.75,0.85,0.55,0.65]},
    {"id":"v20","label":"marathon","category":"Sports","vector":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.7,0.9,0.6,0.7]},
]

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
db=VectorDB();doc_db=DocumentDB()
for v in DEMO: db.insert(v)

class InsertItem(BaseModel):
    id: Optional[str]=None;label:str;category:Optional[str]="custom";vector:list[float]
class DocInsert(BaseModel):
    title:str;text:str
class AskReq(BaseModel):
    question:str;k:int=3

@app.get("/search")
def search(v:str=Query(...),k:int=5,metric:str="cosine",algo:str="hnsw"):
    try: q=[float(x) for x in v.split(",")]
    except: raise HTTPException(400,"Bad vector")
    return db.search(q,k,metric,algo)

@app.post("/insert")
def insert(item:InsertItem):
    d=item.model_dump()
    if not d.get("id"): d["id"]=str(uuid.uuid4())
    db.insert(d);return {"ok":True,"id":d["id"]}

@app.delete("/delete/{id}")
def delete(id:str): db.delete(id);return {"ok":True}

@app.get("/items")
def items(): return {"items":db.items,"count":len(db.items)}

@app.get("/benchmark")
def benchmark(v:str=Query(...),k:int=5,metric:str="cosine"):
    try: q=[float(x) for x in v.split(",")]
    except: raise HTTPException(400,"Bad vector")
    return db.benchmark(q,k,metric)

@app.get("/hnsw-info")
def hnsw_info(): return db.hnsw.info()

@app.get("/stats")
def stats(): return {"demo_vectors":len(db.items),"doc_chunks":len(doc_db.chunks)}

@app.get("/status")
def status(): return ollama_status()

@app.post("/doc/insert")
def doc_insert(body:DocInsert):
    ids=doc_db.insert(body.title,body.text)
    if not ids: raise HTTPException(503,"Ollama offline")
    return {"ok":True,"chunk_ids":ids,"chunks_inserted":len(ids)}

@app.get("/doc/list")
def doc_list(): return {"chunks":doc_db.list(),"count":len(doc_db.chunks)}

@app.delete("/doc/delete/{id}")
def doc_delete(id:str): doc_db.delete(id);return {"ok":True}

@app.post("/doc/ask")
def doc_ask(body:AskReq):
    qv=ollama_embed(body.question)
    if qv is None: raise HTTPException(503,"Ollama offline")
    results=doc_db.search(qv,body.k)
    if not results: return StreamingResponse(iter(["No documents found."]),media_type="text/plain")
    ctx=[]
    for r in results:
        i=r["item"];ctx.append({"id":i["id"],"title":i["title"],"chunk_index":i["chunk_index"],"text":i["text"],"distance":r["distance"]})
    prompt=f"Answer ONLY from context below.\n\nCONTEXT:\n"+"\n\n---\n\n".join(f"[{c['title']} chunk {c['chunk_index']}]\n{c['text']}" for c in ctx)+f"\n\nQUESTION: {body.question}\n\nANSWER:"
    def gen():
        yield "CONTEXT_JSON:"+json.dumps(ctx)+"\n"
        for t in ollama_generate(prompt): yield t
    return StreamingResponse(gen(),media_type="text/plain")

@app.get("/", response_class=HTMLResponse)
def frontend():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(p):
        return open(p, encoding='utf-8').read()
    return "<h1>Place index.html here</h1>"

if __name__=="__main__":
    print("=== VectorDB Python ===")
    print("http://localhost:8080")
    uvicorn.run(app,host="0.0.0.0",port=8080)