
---

# ⭐ PART 1 — Create your AWS GPU machine (EC2)

### 1️⃣ Go to AWS Console

[https://console.aws.amazon.com/](https://console.aws.amazon.com/)

### 2️⃣ Go to:

**EC2 → Instances → Launch Instance**

### 3️⃣ Name it something simple

`hope-training-gpu`

### 4️⃣ Choose OS

**Ubuntu 22.04 LTS (recommended)**

### 5️⃣ Choose instance type

Here pick ONE of these:

### ✔ Best value (recommended)

* **g6e.xlarge** (L40S GPU, 48GB VRAM)

OR

### ✔ Maximum performance

* **p4d.24xlarge** (8× A100, expensive, overkill)
* **p5.48xlarge** (H100, extremely expensive)

Pick the **g6e.xlarge** unless you are rich.

### 6️⃣ Storage

* Size: **200 GB gp3**

### 7️⃣ Create / download SSH key pair

You’ll get a `.pem` file — **DON’T LOSE THIS**.

### 8️⃣ Launch the instance

Wait ~1 minute for it to initialize.

---

# ⭐ PART 2 — Connect to your server (terminal)

On your local computer:

1. Move the key to a safe place (example):

```
~/aws_keys/mykey.pem
```

2. Restrict permissions:

```
chmod 600 ~/aws_keys/mykey.pem
```

3. Connect using SSH:

```
ssh -i ~/aws_keys/mykey.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

Your terminal should now show:

```
ubuntu@ip-xx-xx-xx-xx:~$
```

You are inside your GPU server now.

---

# ⭐ PART 3 — Install necessary software (copy/paste)

Paste these in order:

### 1️⃣ Update system

```
sudo apt update && sudo apt upgrade -y
```

### 2️⃣ Install essentials

```
sudo apt install -y git wget python3 python3-pip python3-venv
```

### 3️⃣ Install NVIDIA drivers (L40S already has correct ones)

```
sudo ubuntu-drivers install
sudo reboot
```

**Reconnect SSH after reboot**

### 4️⃣ Create Python virtual environment

```
python3 -m venv hope_env
source hope_env/bin/activate
```

### 5️⃣ Install PyTorch with CUDA 12

```
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6️⃣ Install HuggingFace tools

```
pip install datasets transformers accelerate sentencepiece
```

### 7️⃣ Install tiktoken (your tokenizer)

```
pip install tiktoken
```

---

# ⭐ PART 4 — Upload your project to the server

Two easy options:

---

## 📌 OPTION A — Upload via GitHub (recommended)

On EC2:

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

## 📌 OPTION B — Upload directly from your computer

From your local terminal:

```
scp -i ~/aws_keys/mykey.pem -r /path/to/your/project ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/
```

This copies your entire project to the EC2 machine.

---

# ⭐ PART 5 — Run your training

Make sure you’re inside the project directory:

```
cd /home/ubuntu/YOUR_PROJECT/Train_model
```

Activate your environment:

```
source ~/hope_env/bin/activate
```

Start training:

```
python train.py
```

You should immediately see logs like:

```
Using device: cuda
Step 0 | loss: ...
Compiling model...
Streaming dataset initialized...
```

---

# ⭐ PART 6 — Monitor GPU usage

Open another terminal window and run:

```
ssh -i ~/aws_keys/mykey.pem ubuntu@YOUR_EC2_IP
watch -n 1 nvidia-smi
```

You should see:

* GPU at 90–100% usage
* VRAM around 20–40GB
* Temperature 70–80°C

This means training is running correctly.

---

# ⭐ PART 7 — Save your checkpoints

Your training script already saves checkpoints (if implemented).
You can download them anytime with:

```
scp -i ~/aws_keys/mykey.pem ubuntu@YOUR_EC2_IP:/home/ubuntu/YOUR_PROJECT/Train_model/checkpoints/* .
```

---

# ⭐ PART 8 — VERY IMPORTANT — Stop the instance when done

Otherwise AWS keeps charging money.

In AWS Console:

### EC2 → Instances → select instance → Actions → **Stop**

**Stop** = safe
**Terminate** = deletes disk + data

If you want checkpoints kept forever → **STOP**, don’t terminate.

---
