#!/usr/bin/env python3
"""Four-Jetson PP1 -> stage-bypass -> PP2 experiment orchestrator.

This process never participates in inference.  It uses root UDS status as the
application truth, SSH only for remote commands, and records every action in
an append-only events.jsonl file.  --dry-run prints the state-machine actions
without connecting to a host.
"""
import argparse, json, os, pathlib, subprocess, sys, time
from datetime import datetime, timezone

NODES = {"root": "192.168.137.13", "node15": "192.168.137.15",
         "node18": "192.168.137.18", "node16": "192.168.137.16"}

def applied_pp(status, source, target):
    pp=status.get("ppMigration",{})
    return (pp.get("appliedFromNodeIndex")==source and pp.get("appliedToNodeIndex")==target and
            isinstance(pp.get("appliedLayers"),list) and bool(pp.get("appliedLayers")))

def verified_bypass(status, generation):
    b=status.get("stageBypass",{})
    return (b.get("rootApplyGeneration")==generation and b.get("verifiedGeneration",0)>=generation and
            b.get("activeStageChain")==[0,1,3])

def netem_apply(interface, peer, ifb, netem):
    return ("sudo tc qdisc replace dev {i} clsact; sudo ip link add {f} type ifb 2>/dev/null || true; "
            "sudo ip link set {f} up; sudo tc qdisc replace dev {f} root netem {n}; "
            "sudo tc filter replace dev {i} egress protocol ip pref 4913 flower dst_ip {p}/32 "
            "action mirred egress redirect dev {f}; sudo tc filter show dev {i} egress").format(i=interface,p=peer,f=ifb,n=netem)

def netem_cleanup(interface, ifb):
    return ("sudo tc filter del dev {i} egress protocol ip pref 4913 2>/dev/null || true; "
            "sudo tc qdisc del dev {f} root 2>/dev/null || true; sudo ip link del {f} 2>/dev/null || true").format(i=interface,f=ifb)

def utc(): return datetime.now(timezone.utc).isoformat()
def uds(sock, req, timeout=2.0):
    import socket
    data = (json.dumps(req, separators=(",", ":")) + "\n").encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout); s.connect(sock); s.sendall(data); buf = b""
        while b"\n" not in buf:
            got = s.recv(65536)
            if not got: break
            buf += got
    return json.loads(buf.split(b"\n", 1)[0].decode())

class Run:
    def __init__(self, a):
        self.a=a; self.dir=pathlib.Path(a.output_dir)/("run_"+a.run_id); self.dir.mkdir(parents=True,exist_ok=False)
        (self.dir/"injections").mkdir(); (self.dir/"uds").mkdir(); self.seq=1
        self.manifest={"run_id":a.run_id,"started_utc":utc(),"nodes":NODES,"uds_socket":a.socket,
          "positions":{"p1":a.p1,"p2":a.p2,"p3":a.p3,"p4":a.p4},"dry_run":a.dry_run,
          "root_cli":a.root_cli,"worker_cli":a.worker_cli,"environment":dict(os.environ)}
        (self.dir/"manifest.json").write_text(json.dumps(self.manifest,indent=2,sort_keys=True))
    def event(self, event_id, phase, status, detail=None, pos=None):
        row={"event_id":event_id,"phase":phase,"status":status,"wall_time_utc":utc(),"root_token_pos":pos,"action_seq":self.seq,"details":detail or {}}
        with (self.dir/"events.jsonl").open("a") as f: f.write(json.dumps(row,sort_keys=True)+"\n")
    def ssh(self, host, command, log):
        argv=["ssh", self.a.ssh_options, host, command] if self.a.ssh_options else ["ssh",host,command]
        if self.a.dry_run: self.event("ssh", "DRY_RUN", "issued", {"host":host,"command":command}); return
        p=subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (self.dir/log).write_text(p.stdout)
        if p.returncode: raise RuntimeError("SSH failed: %s" % host)
    def status(self):
        s=uds(self.a.socket,{"op":"status"});
        with (self.dir/"uds"/"status_samples.jsonl").open("a") as f: f.write(json.dumps({"utc":utc(),"status":s})+"\n")
        return s
    def wait(self, predicate, deadline, label):
        if self.a.dry_run: self.event(label,"DRY_RUN","verified"); return {"position":0}
        end=time.monotonic()+deadline
        while time.monotonic()<end:
            s=self.status()
            if predicate(s): return s
            time.sleep(self.a.poll_seconds)
        raise TimeoutError(label)
    def inject_compute(self, host, tag):
        cmd="sudo scripts/jetson_freq_inject.sh --level 0 --target gpu,cpu --duration %d" % self.a.compute_duration
        self.ssh(host,"nohup %s > /tmp/%s.log 2>&1 & echo $!"%(cmd,tag),"injections/%s.log"%tag)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--socket",default="/tmp/dllama_plan_pp_auto.sock"); p.add_argument("--run-id",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--p1",type=int,required=True); p.add_argument("--p2",type=int,required=True); p.add_argument("--p3",type=int,required=True); p.add_argument("--p4",type=int,required=True)
    p.add_argument("--network-interface",required=True); p.add_argument("--netem",default="delay 200ms 30ms loss 5%"); p.add_argument("--compute-duration",type=int,default=120); p.add_argument("--timeout",type=int,default=180); p.add_argument("--poll-seconds",type=float,default=0.5)
    p.add_argument("--ssh-options",default=""); p.add_argument("--root-cli",default=""); p.add_argument("--worker-cli",default=""); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(); r=Run(a)
    try:
        s=r.wait(lambda x:x.get("position",0)>=a.p1,a.timeout,"baseline"); r.event("compute18_start","COMPUTE_18_ACTIVE","issued",pos=s.get("position")); r.inject_compute(NODES["node18"],"compute_node18")
        g1=s.get("ppMigration",{}).get("appliedGeneration",0); s=r.wait(lambda x:x.get("ppMigration",{}).get("appliedGeneration",0)>g1 and applied_pp(x,2,1),a.timeout,"pp1_apply"); r.event("pp1_apply","PP1_APPLIED_AND_VERIFIED","applied",s.get("ppMigration"),s.get("position"))
        r.wait(lambda x:x.get("position",0)>=a.p2,a.timeout,"network_position")
        for host,peer,ifb,tag in ((NODES["root"],NODES["node18"],"evifb13to18","network_root_before.txt"),(NODES["node18"],NODES["root"],"evifb18to13","network_node18_before.txt")):
            r.ssh(host,netem_apply(a.network_interface,peer,ifb,a.netem),"injections/"+tag)
        req={"op":"set_stage_bypass","cmd":{"seq":r.seq,"mode":"next_barrier","ejectStageIndex":2,"targetStageIndex":3}}; r.seq+=1
        if not a.dry_run: resp=uds(a.socket,req); (r.dir/"uds"/"bypass_response.json").write_text(json.dumps(resp,indent=2));
        r.event("bypass_issue","NETWORK_18_ACTIVE","issued",req)
        gb=s.get("stageBypass",{}).get("rootApplyGeneration",s.get("stageBypass",{}).get("appliedGeneration",0)); s=r.wait(lambda x:x.get("stageBypass",{}).get("rootApplyGeneration",x.get("stageBypass",{}).get("appliedGeneration",0))>gb and x["stageBypass"].get("ejectedStage")==2 and x["stageBypass"].get("targetStage")==3,a.timeout,"bypass_apply"); r.event("bypass_apply","BYPASS_18_APPLIED","applied",s.get("stageBypass"),s.get("position")); bg=s["stageBypass"].get("rootApplyGeneration",s["stageBypass"].get("appliedGeneration")); s=r.wait(lambda x:verified_bypass(x,bg),a.timeout,"bypass_verify"); r.event("bypass_verify","BYPASS_18_VERIFIED","verified",s.get("stageBypass"),s.get("position"))
        r.wait(lambda x:x.get("position",0)>=a.p3,a.timeout,"compute16_position"); r.event("compute16_start","COMPUTE_16_ACTIVE","issued",pos=s.get("position")); r.inject_compute(NODES["node16"],"compute_node16")
        g2=s.get("ppMigration",{}).get("appliedGeneration",0); s=r.wait(lambda x:x.get("ppMigration",{}).get("appliedGeneration",0)>g2 and applied_pp(x,3,1),a.timeout,"pp2_apply"); r.event("pp2_apply","PP2_APPLIED_AND_VERIFIED","applied",s.get("ppMigration"),s.get("position"))
    except Exception as e: r.event("run","FAILED","failed",{"error":str(e)}); return 2
    finally:
        for host,ifb in ((NODES["root"],"evifb13to18"),(NODES["node18"],"evifb18to13")):
            try: r.ssh(host,netem_cleanup(a.network_interface,ifb),"injections/network_cleanup.txt")
            except Exception: pass
    r.event("finish","DRAIN_AND_RESTORE","verified"); return 0
if __name__=="__main__": sys.exit(main())
