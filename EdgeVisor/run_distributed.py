import paramiko
import time
import threading
import sys

# Configuration
# Workers list as per instructions
WORKERS = [
    {"host": "192.168.182.11", "user": "cc", "password": "cc123"},
    {"host": "192.168.182.12", "user": "cc", "password": "cc123"},
    {"host": "192.168.182.13", "user": "cc", "password": "cc123"},
    {"host": "192.168.182.15", "user": "jetson", "password": "yahboom"},
    {"host": "192.168.182.17", "user": "jetson", "password": "yahboom"},
]

# Root/Master node
MASTER = {"host": "192.168.182.16", "user": "jetson", "password": "yahboom"}

# Common project path
PROJECT_PATH = "~/yanhui/distributed-llama"
WORKER_CMD = f"cd {PROJECT_PATH} && ./dllama worker --port 9999 --nthreads 4"

# Master command with specified worker order: 15, 11, 12, 13, 17
MASTER_CMD = (
    f"cd {PROJECT_PATH} && ./dllama inference "
    "--model /home/models/dllama_model_qwen3_8b_q40.m "
    "--tokenizer /home/models/dllama_tokenizer_qwen3_8b_q40.t "
    "--buffer-float-type q80 "
    "--benchmark "
    "--nthreads 4 "
    "--prompt \"The capital of France is\" "
    "--steps 128 "
    "--workers 192.168.182.15:9999 192.168.182.11:9999 192.168.182.12:9999 192.168.182.13:9999 192.168.182.17:9999 "
    "--ratios \"1:1@20*3:3:2@8*1@8\""
)

def create_ssh_client(host, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password, timeout=10)
        return client
    except Exception as e:
        print(f"Failed to connect to {host}: {e}")
        return None

def run_command_stream(client, command, prefix=""):
    """Runs a command and streams output to stdout."""
    try:
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        
        # Stream output line by line
        for line in iter(stdout.readline, ""):
            print(f"{prefix}{line}", end="")
            
        exit_status = stdout.channel.recv_exit_status()
        return exit_status
    except Exception as e:
        print(f"{prefix}Error running command: {e}")
        return -1

def start_worker(worker_config, stop_event):
    """Starts a worker process and keeps it running until stop_event is set."""
    host = worker_config["host"]
    user = worker_config["user"]
    password = worker_config["password"]
    
    print(f"[{host}] Connecting...")
    client = create_ssh_client(host, user, password)
    if not client:
        return

    try:
        # First, kill any existing dllama processes to ensure a clean state
        client.exec_command("pkill -f dllama")
        time.sleep(1) # Wait for cleanup
        
        print(f"[{host}] Starting worker: {WORKER_CMD}")
        stdin, stdout, stderr = client.exec_command(WORKER_CMD, get_pty=True)
        
        while not stop_event.is_set():
            if stdout.channel.exit_status_ready():
                print(f"[{host}] Worker exited unexpectedly!")
                break
            # Read line if available
            if stdout.channel.recv_ready():
                output = stdout.channel.recv(1024).decode('utf-8', errors='ignore')
                # print(f"[{host}] {output}", end="") # Optional: print worker output
            time.sleep(0.1)
            
    except Exception as e:
        print(f"[{host}] Error: {e}")
    finally:
        print(f"[{host}] Stopping worker...")
        try:
            stdin.write('\x03') # Ctrl+C
            time.sleep(1)
            client.exec_command("pkill -f dllama")
        except:
            pass
        client.close()
        print(f"[{host}] Disconnected.")

def main():
    threads = []
    stop_event = threading.Event()
    
    print("--- Starting Distributed Llama Inference ---")
    
    # 1. Start Workers
    print("\n>>> Starting Workers...")
    for worker in WORKERS:
        t = threading.Thread(target=start_worker, args=(worker, stop_event))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait a bit for workers to initialize
    time.sleep(5)
    
    # 2. Run Master
    print("\n>>> Starting Master Node...")
    master_client = create_ssh_client(MASTER["host"], MASTER["user"], MASTER["password"])
    if master_client:
        try:
            print(f"[{MASTER['host']}] Executing inference...")
            # Run cleanup on master too, just in case
            master_client.exec_command("pkill -f dllama")
            time.sleep(1)
            
            # Run the inference command
            exit_code = run_command_stream(master_client, MASTER_CMD, prefix=f"[{MASTER['host']}] ")
            
            if exit_code == 0:
                print("\n>>> Inference completed successfully.")
            else:
                print(f"\n>>> Inference failed with exit code {exit_code}.")
                
        finally:
            master_client.close()
    else:
        print("Failed to connect to master node.")
        
    # 3. Cleanup
    print("\n>>> Stopping Workers...")
    stop_event.set()
    for t in threads:
        t.join(timeout=2)
        
    print("--- Done ---")

if __name__ == "__main__":
    main()
