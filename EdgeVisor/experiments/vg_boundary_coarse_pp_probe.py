#!/usr/bin/env python3
"""Extra low-bandwidth probe for VG boundary."""
import csv, json, time
from dataclasses import asdict, dataclass
from pathlib import Path
import edgevisor_ablation_suite as suite
RUN_ID = time.strftime('vg_boundary_coarse_%Y%m%d_%H%M%S')
suite.RUN_ID = RUN_ID
suite.OUT = suite.BASE / 'vg_boundary_results' / RUN_ID
suite.SOCK_DIR = suite.OUT / 's'
OUT = suite.OUT
MODEL='3B'; STEPS=6; MAX_SEQ_LEN=1024; BANDWIDTH_MBPS=1; CPUS=[2.0]*8
TOPOLOGIES=[
 ('vg_332_all8','1:1:1*1:1:1*1:1',7,'all 8 nodes, 3/3/2 VG with TP inside stages'),
 ('pp_8stage_all8','1@4*1@4*1@4*1@4*1@3*1@3*1@3*1@3',7,'all 8 nodes, no VG, singleton PP stages'),
 ('pp_3stage_strong','1@10*1@9*1@9',2,'coarse PP fallback using only root + 2 strong workers'),
]
@dataclass
class Row:
    run_id:str; topology:str; ratios:str; active_workers:int; note:str; bandwidth_mbps:int; steps:int; root_exit:int; wall_ms:float; avg_pred_ms:float; effective_tokens_s:float; output_dir:str
rows=[]
def write_outputs():
    OUT.mkdir(parents=True, exist_ok=True)
    if rows:
        fields=list(asdict(rows[0]).keys())
        with (OUT/'runs.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow(asdict(r)) for r in rows]
    by={r.topology:r for r in rows}; cmp=[]; base=by.get('vg_332_all8')
    if base:
        for r in rows:
            if r.topology==base.topology: continue
            cmp.append({'baseline':base.topology,'candidate':r.topology,'baseline_wall_ms':round(base.wall_ms,3),'candidate_wall_ms':round(r.wall_ms,3),'candidate_minus_baseline_wall_ms':round(r.wall_ms-base.wall_ms,3),'baseline_tokens_s':round(base.effective_tokens_s,5),'candidate_tokens_s':round(r.effective_tokens_s,5),'candidate_vs_baseline_tokens_s_delta_pct':round(((r.effective_tokens_s-base.effective_tokens_s)/base.effective_tokens_s*100.0),3) if base.effective_tokens_s else '', 'verdict':'candidate_PP_better' if r.wall_ms<base.wall_ms else 'VG_better'})
    if cmp:
        fields=list(cmp[0].keys())
        with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(cmp)
    (OUT/'summary.json').write_text(json.dumps({'run_id':RUN_ID,'output':str(OUT),'rows':[asdict(r) for r in rows],'comparisons':cmp,'design':{'bandwidth_mbps':BANDWIDTH_MBPS,'steps':STEPS,'topologies':TOPOLOGIES}},indent=2),encoding='utf-8')
    lines=['# VG Boundary Coarse PP Probe','',f'- Run: `{RUN_ID}`',f'- Bandwidth: `{BANDWIDTH_MBPS} Mbps`',f'- Steps: `{STEPS}`','','## Summary','']
    if cmp:
        lines += ['| candidate | cand-base wall ms | base tok/s | cand tok/s | delta % | verdict |','| --- | ---: | ---: | ---: | ---: | --- |']
        for c in cmp: lines.append(f"| {c['candidate']} | {c['candidate_minus_baseline_wall_ms']} | {c['baseline_tokens_s']} | {c['candidate_tokens_s']} | {c['candidate_vs_baseline_tokens_s_delta_pct']} | {c['verdict']} |")
    (OUT/'vg_boundary_coarse_pp_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def run_one(topology, ratios, active_workers, note):
    suite.RATIOS=ratios; suite.ACTIVE_WORKER_COUNT=active_workers; suite.BASE_NET_MBPS=[BANDWIDTH_MBPS]*8; suite.CPU_LIMITS=CPUS; suite.ROOT_THREADS=2; suite.WORKER_THREADS=2
    meta=suite.start_cluster(MODEL,topology,'vg_boundary_coarse',steps=STEPS,max_seq_len=MAX_SEQ_LEN,extra_args='',prompt='The capital of France is')
    t0=time.time(); root_exit=999
    try:
        wait=suite.run(['docker','wait',meta['root']],check=False,timeout=1200)
        root_exit=int((wait.stdout or '999').strip().splitlines()[-1]) if (wait.stdout or '').strip() else 999
        wall_ms=(time.time()-t0)*1000.0; root_log=suite.read_logs(meta['root'])
        _eval_ts,pred_ts,_post_ts,avg_pred,_rejects=suite.parse_root_metrics(root_log)
        eff=pred_ts if pred_ts else (1000.0/avg_pred if avg_pred else 0.0)
        rows.append(Row(RUN_ID,topology,ratios,active_workers,note,BANDWIDTH_MBPS,STEPS,root_exit,wall_ms,avg_pred,eff,meta['test_dir']))
        write_outputs()
    finally:
        suite.stop_cluster(meta); write_outputs()
def main():
    OUT.mkdir(parents=True,exist_ok=True); suite.SOCK_DIR.mkdir(parents=True,exist_ok=True); suite.cleanup_prefix('b01ab_'); print(f'OUT={OUT}',flush=True)
    try:
        for topo in TOPOLOGIES:
            print('RUN',topo[0],flush=True); run_one(*topo)
    finally:
        suite.cleanup_prefix('b01ab_'); write_outputs()
    print(f'DONE OUT={OUT}',flush=True)
if __name__=='__main__': main()
