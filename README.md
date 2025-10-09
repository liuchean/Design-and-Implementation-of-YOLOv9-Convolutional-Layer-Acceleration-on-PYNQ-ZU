# Design and Implementation of YOLOv9 Convolutional Layer Acceleration on PYNQ-ZU

## 📘 專案簡介
深度學習（Deep Learning）與人工智慧（Artificial Intelligence, AI）近年來已成為科技發展的核心。  
其中，影像識別技術因具備廣泛應用價值而備受關注。卷積神經網路（Convolutional Neural Network, CNN）是影像辨識的主要架構，而卷積運算則為整體運算中最關鍵且最耗時的部分。  
由於卷積運算涉及大量重複且複雜的資料處理，傳統以 ARM 或 x86 為主的處理器架構常面臨效能瓶頸。  
FPGA（Field Programmable Gate Array）具備高度並行運算特性，能有效加速此類運算流程，顯著提升邊緣運算效能。

---

## 🧠 研究背景
卷積神經網路在影像辨識中扮演關鍵角色，但由於其運算量龐大，常導致一般 CPU 或 GPU 架構在邊緣端部署時受限於能耗與延遲。  
本研究以 **PYNQ-ZU（FPGA）** 為硬體平台，探討如何透過結構性平行化與資料流（Dataflow）優化，有效提升 YOLOv9 模型卷積層的運算效率與穩定度。

---

## 🧩 實作與開發環境
- **硬體平台**：Xilinx PYNQ-ZU (Zynq UltraScale+)  
- **開發工具**：Vitis HLS、Vivado、PYNQ Framework  
- **開發語言**：C / C++（硬體架構）、Python（控制與驗證）  

---

## ⚙️ 主要系統架構設計

### 🚀 EPIC（Enhanced Partition for Independent Convolution）架構設計背景
在 PYNQ-ZU 平台上若直接執行卷積運算而未採用最佳化策略，將導致 FPGA 無法充分發揮並行運算優勢。  
為此，本研究提出 **EPIC 架構**，透過輸入與權重資料分割及多模組平行運算，達到更高的資料吞吐量與運算效率。

### 🧱 EPIC 架構說明
EPIC 架構將原先單一大型卷積模組拆分為 **四個可獨立運作的小型卷積模組**，使各模組能同時運算並互不干擾。  
同時，透過資料流（Dataflow）技術與可調整參數設計，實現更靈活的運算架構與高效能並行化。

#### 架構特點：
- **輸入與權重分割**：  
  將資料分割為兩部分以減少記憶體存取衝突。  
- **多運算模組平行化**：  
  使用四個 `Conv` 模組同步運算，提升整體資料吞吐量。  
- **資料流優化 (Dataflow)**：  
  使資料傳輸與運算重疊，降低延遲。  
- **參數彈性化設計**：  
  可根據不同卷積層動態調整參數設定。

#### EPIC 架構示意：
<img width="500" height="400" alt="EPIC Architecture" src="https://github.com/user-attachments/assets/a7e12ce5-4ef9-4228-8b46-3ab39c02b87e" />

#### Parallel Processing 模組：
<img width="500" height="293" alt="Parallel Processing" src="https://github.com/user-attachments/assets/acef31b0-70e0-4153-83b2-d67d63b79ffa" />

#### 模組功能總覽：

| 模組名稱 | 功能描述 |
|-----------|-----------|
| `EPIC` | 管理整體卷積運算流程 |
| `Parallel Processing` | 核心平行運算模組 |
| `Conv` | 卷積核心運算單元 |
| `Output Add` | 負責整合四個 Conv 模組的結果 |

---

## 🔧 硬體資源優化設計

### 1️⃣ 卷積 IP 共用設計

#### ✳️ 共用動機
FPGA 中的 IP（Intellectual Property Core）相當於硬體的功能模組。  
不同卷積參數會導致硬體佈局差異，若每種參數都建立獨立 IP，會造成資源浪費。  
因此本研究提出 **IP 共用策略**，透過模板化的設計使多種卷積設定可共用同一硬體架構。

#### 🧩 共用策略
根據卷積運算中不同參數組合的使用頻率，挑選具代表性的參數模板，建立 **TIP (Template IP)**。  
同時於 Vitis HLS 中擴展 EPIC 設計，使多個卷積任務可共用同一 TIP，節省硬體資源。

#### 🧠 共用架構示意：
在原有 EPIC 架構中引入 Parallel Processing 模組與運算規模判斷模組（Check），  
根據不同的卷積規模動態分配計算資源，避免小型卷積浪費大模組資源。

<img width="400" height="430" alt="TIP Architecture" src="https://github.com/user-attachments/assets/073ff877-eb9e-4ffc-872c-a5daa9f02dc6" />

---

### 2️⃣ 儲存精度優化

#### 🎯 問題背景
原始 EPIC 架構採用 `float32` 格式進行卷積運算，導致 BRAM（Block RAM）資源佔用率高達 **86.11%**，  
成為硬體擴展的主要瓶頸。

<img width="400" height="328" alt="BRAM Usage" src="https://github.com/user-attachments/assets/08af7a38-e07c-4046-8b2a-ac75433b045e" />

#### 🔍 資源瓶頸分析
- 高維度特徵圖以 32-bit 格式儲存，造成 BRAM 消耗極大。  
- YOLOv9 模型的特徵圖尺寸龐大，進一步加劇記憶體壓力。

#### 💡 優化策略
採用 `ap_fixed<18,6>`（18-bit 固定小數格式）取代 `float32`，  
有效降低 BRAM 使用率並保持 **98% 的原始辨識精度**。  

> - **總位寬**：18 bit  
> - **整數位元**：6 bit（含符號位）  
> - **小數位元**：12 bit  
> - **精度表現**：辨識率維持約 98%，資源利用率提升 3 倍。

---

## ⚡ 效能比較結果

| 指標項目 | FPGA (PYNQ-ZU) | ARM Cortex-A53 | 提升幅度 |
|-----------|----------------|----------------|-----------|
| **執行時間** | 530.37 ms | 241.99 ms | 執行時間降低 **54.37%** |
| **運算穩定度（標準差）** | 0.11 | 2.48 | 運算穩定性大幅提升 |

> 🔹 本架構在資源優化後，實現三倍的硬體資源利用率 
> 🔹 保持原模型約 **98% 的推論準確率**

---

## 🧾 研究論文

> 劉哲安，《基於 FPGA 之 YOLOv9 平行化加速架構設計與實作》，  
> 國立台北大學通訊工程學系碩士論文，2025。

### 🎯 研究重點
- 針對 YOLOv9 模型中卷積層的高運算量問題，設計適用於 FPGA 的加速架構。

### 🧩 技術貢獻
- 結合 C/C++ 與 Python 進行軟硬體協同設計。  
- 在有限硬體資源下完成多層次最佳化（運算單元配置、記憶體策略）。  
- 完成實際 FPGA 部署與效能驗證，確保穩定性與可行性。

---

⭐ **本研究展示了 FPGA 在深度學習推論加速上的潛力，並證明透過架構級優化可在有限資源下達成高效能運算。**
