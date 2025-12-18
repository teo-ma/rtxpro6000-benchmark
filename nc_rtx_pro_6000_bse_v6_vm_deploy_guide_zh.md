# NC RTX Pro 6000 BSE v6（NCv6）GPU VM 部署指南（中文）

> 基于《NC RTX PRO 6000 BSE v6 Virtual Machines – Public Preview Onboarding Guide》（2025-12-16，v0.3）中“GPU VM 部署方法”整理。
>
> 目标：用最少步骤把 NCv6（RTX Pro 6000 Blackwell Server Edition）GPU VM 在 Azure 上部署起来，并完成驱动安装与验证。

## 1. 适用范围与重要说明

- **适用阶段**：本文面向 **Public Preview**（公测预览）阶段的 NCv6。
- **访问门槛**：NCv6 预览访问是 **gated（需 allowlist）** 的；未被放行的订阅即使执行部署命令也可能找不到 SKU/无法创建。
- **部署入口限制**：在早期预览阶段，**Azure Portal 可能看不到这些规格**；按指南要求优先使用 **Azure CLI / PowerShell** 部署。
- **VM 镜像支持**：文档说明当前支持 **Windows/Linux** 常规镜像；但 **Azure HPC/AI VM images（Ubuntu-HPC、AlmaLinux-HPC）暂不支持**。

## 2. 前置条件（Prerequisites）

1. **订阅已 allowlist**：先按公测流程提交访问申请表并等待放行。
2. **已安装并登录 Azure CLI**：本地/跳板机可运行 `az login`。
3. **资源组已创建**：例如 `az group create -n <rg> -l <region>`。
4. **区域与配额**：确保订阅在目标 Region 具备对应 GPU VM 配额（若创建失败，优先检查配额与 allowlist 状态）。

## 3. 规格（Size）与区域（Region）可用性

### 3.1 当前可用 NCv6 规格（文档列出的可测试 SKU）

- `Standard_NC128ds_xl_RTXPRO6000BSE_v6`
- `Standard_NC256ds_xl_RTXPRO6000BSE_v6`
- `Standard_NC320ds_xl_RTXPRO6000BSE_v6`
- `Standard_NC128lds_xl_RTXPRO6000BSE_v6`
- `Standard_NC256lds_xl_RTXPRO6000BSE_v6`
- `Standard_NC320lds_xl_RTXPRO6000BSE_v6`

### 3.2 区域可用性（文档给出的时间点）

- **West US 2**：2025-12-16 起可用
- **North Europe**：2026-01-19 起可用
- **Southeast Asia**：2026-01-19 起可用

## 4. 部署步骤（Azure CLI）

> 文档提供的是命令行部署示例。下面给出可直接套用的“最小可用”流程。

### 4.1 变量准备

```bash
RG="<your-resource-group>"
VM="<your-vm-name>"
LOC="westus2"  # 例如 westus2
SIZE="Standard_NC128ds_xl_RTXPRO6000BSE_v6"
IMAGE="Ubuntu2404"
ADMIN="<admin-username>"
```

### 4.2 创建 VM（Linux 示例）

文档示例使用 Ubuntu 24.04（`Ubuntu2404`）与用户名/密码参数：

```bash
az vm create \
  --resource-group "$RG" \
  --name "$VM" \
  --size "$SIZE" \
  --image "$IMAGE" \
  --admin-username "$ADMIN" \
  --admin-password "<admin-password>" \
  --location "$LOC"
```

建议（可选）：生产环境更推荐使用 SSH key（避免明文密码），例如：

```bash
az vm create \
  --resource-group "$RG" \
  --name "$VM" \
  --size "$SIZE" \
  --image "$IMAGE" \
  --admin-username "$ADMIN" \
  --authentication-type ssh \
  --generate-ssh-keys \
  --location "$LOC"
```

### 4.3 Windows 部署（提示）

- 文档说明 **Windows VM images 也支持**。
- 创建方式同上，只需将 `--image` 换成对应 Windows Server 镜像，并按 Windows 远程登录方式配置（如密码）。

## 5. 驱动安装与验证

> 文档给出了 NCv6 上 vGPU Unified Driver（GRID Azure 版本）的安装方式。

### 5.1 Linux：安装 vGPU Unified Driver（文档示例）

1) 安装依赖并更新：

```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y build-essential
```

2) 下载并安装驱动（文档链接与版本号）：

```bash
wget https://download.microsoft.com/download/85beffdc-8361-4df4-a823-dcb1b230a7aa/NVIDIA-Linux-x86_64-580.105.08-grid-azure.run
sudo sh NVIDIA-Linux-x86_64-580.105.08-grid-azure.run -M open
```

3) 验证：

```bash
nvidia-smi
```

### 5.2 Windows：安装 vGPU Unified Driver（文档提示）

- 文档提示下载 **Windows Driver（v19.3 vGPU Unified Driver）** 并按 N 系列 VM 的 Windows 驱动安装文档完成。
- 验证同样可使用 `nvidia-smi`。

### 5.3 验证注意事项

- 文档指出：即使驱动安装成功，`nvidia-smi` 中 **GPU-Util 可能显示为 N/A**，直到你在 VM 上运行一次 GPU 工作负载。

## 6. 常见问题（FAQ 摘要）

### 6.1 Portal 找不到 NCv6 规格

- 文档说明：在初期 Public Preview 阶段，**NCv6 规格可能无法通过 Azure Portal 直接部署**。
- 处理方式：使用 **Azure CLI / PowerShell**。
- 文档预期：这些 SKU 预计在 **2026 年 1 月**开始在 Portal 中可见并可部署（具体以官方进展为准）。

### 6.2 镜像兼容性

- 支持：常规 Windows/Linux 镜像。
- 暂不支持：`Ubuntu-HPC`、`AlmaLinux-HPC` 等 HPC/AI VM images。

## 7. 参考与来源

- Source PDF：`NC RTX PRO 6000 BSE v6 - Public Preview Onboarding Guide.pdf`（2025-12-16，v0.3）
- 本文为流程性总结，**未复刻定价表等原文内容**；如需计费/定价细节请以原 PDF 或 Azure 官方定价页面为准。
