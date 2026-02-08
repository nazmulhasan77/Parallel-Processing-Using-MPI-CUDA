## 🚀 Step-by-Step CUDA Installation on Linux (Ubuntu/Debian)

### 🧰 1) **Prerequisites**

✔ A **CUDA-capable NVIDIA GPU**
✔ You’re on a supported Linux distro (Ubuntu 22.04/24.04 etc.) ([NVIDIA Docs][1])
✔ A compiler toolchain is installed (`build-essential`)
✔ Optional but recommended: latest **NVIDIA driver installed** before CUDA

Check your current NVIDIA driver with:

```bash
nvidia-smi
```

If no driver, install it first:

```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

---

## 🧱 2) Choose an Installation Method

There are two common ways:

### ✅ **Method A: APT Package (easy & recommended)**

This uses NVIDIA’s repository and integrates with `apt`.

**Step A1 — Add CUDA repo keyring**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

**Step A2 — Install CUDA Toolkit**

```bash
sudo apt install cuda-toolkit
```

> This installs `nvcc`, libcuda, math libs, and tools. ([NVIDIA Docs][2])

---

### 🛠️ **Method B: Runfile Installer**

This gives you more control (e.g., skip driver install):

1. Download the `.run` installer from NVIDIA’s CUDA download page.

2. Run:

   ```bash
   sudo sh cuda_<version>_linux.run
   ```

3. In the interactive installer, **deselect the driver** if you already installed it.

4. Reboot when done. ([NVIDIA Docs][1])

---

## 📌 3) **Configure Environment Variables**

Add CUDA to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

This makes tools like `nvcc` and libraries available to the system.

---

## 🧪 4) Verify the Installation

### Check the compiler

```bash
nvcc -V
```

You should see the CUDA version printout.

### Compile sample (optional)

```bash
cuda-install-samples-$(uname -m).sh ~
cd ~/NVIDIA_CUDA-*/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

If you see **PASS**, CUDA is working.

---

## ⚠️ Common Issues & Tips

* **Kernel headers missing:** install `linux-headers-$(uname -r)` if building drivers fails. ([developer.download.nvidia.cn][3])
* **Secure Boot problems:** might block driver modules — disable Secure Boot in BIOS.
* **Ubuntu version mismatch:** choose the correct repository for your Ubuntu version.

---

## 🧠 Summary

| Step          | Command                         |
| ------------- | ------------------------------- |
| Add CUDA repo | `dpkg -i cuda-keyring`          |
| Update        | `sudo apt update`               |
| Install CUDA  | `sudo apt install cuda-toolkit` |
| Set paths     | Add to `.bashrc`                |
| Verify        | `nvcc -V`, `deviceQuery`        |

---
