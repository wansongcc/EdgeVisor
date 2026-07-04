# How to Run Distributed Llama on 🍓 Raspberry Pi

This article describes how to run Distributed Llama on 4 Raspberry Pi devices, but you can also run it on 1, 2, 4, 8... devices. Please adjust the commands and topology according to your configuration.

````
[🔀 SWITCH OR ROUTER]
      | | | |
      | | | |_______ 🔸 raspberrypi1 (ROOT)     10.0.0.1
      | | |_________ 🔹 raspberrypi2 (WORKER 1) 10.0.0.2:9999
      | |___________ 🔹 raspberrypi3 (WORKER 2) 10.0.0.3:9999
      |_____________ 🔹 raspberrypi4 (WORKER 3) 10.0.0.4:9999
````

1. Install `Raspberry Pi OS Lite (64 bit)` on your **🔸🔹 ALL** Raspberry Pi devices. This OS doesn't have desktop environment but you can easily connect via SSH to manage it.
2. Connect **🔸🔹 ALL** devices to your **🔀 SWITCH OR ROUTER** via Ethernet cable. If you're using only two devices, it's better to connect them directly without a switch.
3. Connect to all devices via SSH from your computer.

```
ssh user@raspberrypi1.local
ssh user@raspberrypi2.local
ssh user@raspberrypi3.local
ssh user@raspberrypi4.local
```

4. Install Git on **🔸🔹 ALL** devices:

```sh
sudo apt install git
```

5. Clone this repository and compile Distributed Llama on **🔸🔹 ALL** devices:

```sh
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
make dllama-api
```

6. Download the model to the **🔸 ROOT** device using the `launch.py` script. You don't need to download the model on worker devices.

```sh
python3 launch.py # Prints a list of available models

python3 launch.py llama3_2_3b_instruct_q40 # Downloads the model to the root device
```

7. Assign static IP addresses on **🔸🔹 ALL** devices. Each device must have a unique IP address in the same subnet.

```sh
sudo ip addr add 10.0.0.1/24 dev eth0 # 🔸 ROOT
sudo ip addr add 10.0.0.2/24 dev eth0 # 🔹 WORKER 1
sudo ip addr add 10.0.0.3/24 dev eth0 # 🔹 WORKER 2
sudo ip addr add 10.0.0.4/24 dev eth0 # 🔹 WORKER 3
```

8. Start workers on all **🔹 WORKER** devices:

```sh
sudo nice -n -20 ./dllama worker --port 9999 --nthreads 4
```

9. Run the inference to test if everything works fine on the **🔸 ROOT** device:

```sh
sudo nice -n -20 ./dllama inference \
  --prompt "Hello world" \
  --steps 32 \
  --model models/llama3_2_3b_instruct_q40/dllama_model_llama3_2_3b_instruct_q40.m \
  --tokenizer models/llama3_2_3b_instruct_q40/dllama_tokenizer_llama3_2_3b_instruct_q40.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

10. To run the API server, start it on the **🔸 ROOT** device:

```sh
sudo nice -n -20 ./dllama-api \
  --port 9999 \
  --model models/llama3_2_3b_instruct_q40/dllama_model_llama3_2_3b_instruct_q40.m \
  --tokenizer models/llama3_2_3b_instruct_q40/dllama_tokenizer_llama3_2_3b_instruct_q40.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

Now you can connect to the API server from your computer:

```
http://raspberrypi1.local:9999/v1/models
```

11. When the API server is running, you can open the web chat in your browser, open [llama-ui.js.org](https://llama-ui.js.org/), go to the settings and set the base URL to: `http://raspberrypi1.local:9999`. Press the "save" button and start chatting!

## Online Migration Notes

The Raspberry Pi commands above are the baseline startup path. Current dynamic
migration parameters are documented in:

- [README_ENV_VARS.md](README_ENV_VARS.md)
- [HOW_TO_ONLINE_MIGRATION.md](HOW_TO_ONLINE_MIGRATION.md)
- [UNEVEN_TP_PP_CONFIG.md](UNEVEN_TP_PP_CONFIG.md)

For automatic root-side TPOT scheduling, add
`--enable-dynamic-tpot --plan-ctrl-socket <path>` to root inference. PP layer
migration also requires a PP layout with redundant boundary weights, for example
`--runtime-redundant-boundary-layers 1`.
