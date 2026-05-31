# How to Run Distributed Llama on 💻 Linux, MacOS or Windows

This article describes how to run Distributed Llama on 4 devices, but you can also run it on 1, 2, 4, 8... devices. Please adjust the commands and topology according to your configuration.

````
[🔀 SWITCH OR ROUTER]
      | | | |
      | | | |_______ 🔸 device1 (ROOT)     10.0.0.1
      | | |_________ 🔹 device2 (WORKER 1) 10.0.0.2:9999
      | |___________ 🔹 device3 (WORKER 2) 10.0.0.3:9999
      |_____________ 🔹 device4 (WORKER 3) 10.0.0.4:9999
````

1. Install Git and C++ compiler on **🔸🔹 ALL** devices:

  * Linux: 
    ```
    sudo apt install git build-essential
    ```
  * MacOS
    ```
    brew install git
    ```
  * Windows

    Install Git and Mingw (via [Chocolatey](https://chocolatey.org/install)):
    ```powershell
    choco install git mingw
    ```

2. Connect **🔸🔹 ALL** devices to your **🔀 SWITCH OR ROUTER** via Ethernet cable. If you're using only two devices, it's better to connect them directly without a switch.

3. Clone this repository and compile Distributed Llama on **🔸🔹 ALL** devices:

```sh
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
make dllama-api
```

## Notes for ARM boards (Raspberry Pi / NVIDIA Jetson)

- Recommended: compile *on the target device* (Raspberry Pi OS 64-bit / Jetson Linux aarch64).
- If you need a **portable** binary that can run across different ARM CPUs (e.g. build on one Jetson and run on another Jetson/Pi), build with:
  - `make DLLAMA_PORTABLE=1 dllama`
  - `make DLLAMA_PORTABLE=1 dllama-api`
- For Jetson GPU usage via Vulkan, see [docs/HOW_TO_RUN_GPU.md](./HOW_TO_RUN_GPU.md). When using `--gpu-index`, set `--nthreads 1`.

4. Download the model to the **🔸 ROOT** device using the `launch.py` script. You don't need to download the model on worker devices.

```sh
python3 launch.py # Prints a list of available models

python3 launch.py llama3_2_3b_instruct_q40 # Downloads the model to the root device
```

5. Start workers on all **🔹 WORKER** devices:

```sh
./dllama worker --port 9999 --nthreads 4
```

6. Run the inference to test if everything works fine on the **🔸 ROOT** device:

```sh
./dllama inference \
  --prompt "Hello world" \
  --steps 32 \
  --model models/llama3_2_3b_instruct_q40/dllama_model_llama3_2_3b_instruct_q40.m \
  --tokenizer models/llama3_2_3b_instruct_q40/dllama_tokenizer_llama3_2_3b_instruct_q40.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

7. To run the API server, start it on the **🔸 ROOT** device:

```sh
./dllama-api \
  --port 9999 \
  --model models/llama3_2_3b_instruct_q40/dllama_model_llama3_2_3b_instruct_q40.m \
  --tokenizer models/llama3_2_3b_instruct_q40/dllama_tokenizer_llama3_2_3b_instruct_q40.t \
  --buffer-float-type q80 \
  --nthreads 4 \
  --max-seq-len 4096 \
  --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

Now you can connect to the API server:

```
http://10.0.0.1:9999/v1/models
```

8. When the API server is running, you can open the web chat in your browser, open [llama-ui.js.org](https://llama-ui.js.org/), go to the settings and set the base URL to: `http://10.0.0.1:9999`. Press the "save" button and start chatting!
