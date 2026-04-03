# 8-Device Remote Experiment

Run on remote host (not inside container):

```bash
python3 /home/cc/yhbian/edgevisor_code/examples/edgevisor_ext/experiments/run_8dev_remote.py
```

Defaults:
- compose: `/home/cc/yhbian/docker_file/docker-compose-edgevisor-8dev.yml`
- code dir mount: `/home/cc/yhbian/edgevisor_code`
- model dir mount: `/home/cc/dllama/distributed-llama/models`
- report: `/workspace/examples/edgevisor_ext/phase1_8dev_remote_report.md` (inside `ev8_n0`)

After run, fetch report from host path:

```bash
docker cp ev8_n0:/workspace/examples/edgevisor_ext/phase1_8dev_remote_report.md /tmp/phase1_8dev_remote_report.md
```
