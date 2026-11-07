import json, math
# name, provider, inr_per_hr (portal, on-demand, 1 card), approx peak dense BF16 TFLOPS, memory GB, note
cards = [
 ("H200 SXM","Neysa",140.00,989,141,"Same CUDA stack as H100"),
 ("H200 NVL","Jio Platforms",146.38,835,141,"Lower-power H200"),
 ("B200 SXM","Yotta / Cyfuture",290.70,2250,180,"Underused on small models"),
 ("B300 SXM","Neysa",351.00,2250,288,"Extra speed mostly in FP4"),
 ("MI300X","Jio Platforms",168.20,1307,192,"AMD ROCm software"),
 ("MI325X","Cyfuture",169.20,1307,256,"AMD ROCm software"),
 ("TPU v6e","Sify",114.94,918,32,"Needs JAX / PyTorch-XLA port"),
 ("A100 80GB","E2E Networks",135.90,312,80,"Older generation"),
 ("L4","Sify",44.86,121,24,"Small, no fast links between cards"),
 ("Trainium trn1","Locuz",130.21,190,32,"AWS Neuron software"),
]
# model, params, tokens, assumed MFU, cluster size for wall-clock
models = [
 ("Chotu (181M)",181e6,10e9,0.15,1),
 ("Middle (7B)",7e9,140e9,0.40,64),
 ("Big (100B)",100e9,2e12,0.40,1024),
]
rows=[]
for m,p,t,mfu,n in models:
    flops=6*p*t
    state_gb=16*p/1e9
    for c,prov,price,tf,mem,note in cards:
        gh=flops/(mfu*tf*1e12)/3600
        rows.append(dict(model=m,card=c,provider=prov,inr_per_hr=price,tflops=tf,mem_gb=mem,
            params=p,tokens=t,mfu=mfu,flops=flops,gpu_hours=round(gh,1),cost_inr=round(gh*price),
            cluster=n,days_on_cluster=round(gh/n/24,1),min_cards_to_hold=math.ceil(state_gb/mem),note=note))
json.dump(rows,open('rows.json','w'),indent=1)
for m,*_ in models:
    rs=sorted([r for r in rows if r['model']==m],key=lambda r:r['cost_inr'])
    print("\n",m,"FLOPs=%.2e"%rs[0]['flops'])
    for r in rs: print(f"  {r['card']:14} {r['gpu_hours']:>14,.1f} GPU-h  ₹{r['cost_inr']:>16,}  {r['days_on_cluster']:>8} d on {r['cluster']}  hold:{r['min_cards_to_hold']}")
