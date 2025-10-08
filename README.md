# Design-and-Implementation-of-YOLOv9-Convolutional-Layer-Acceleration-on-PYNQ-ZU

## 專案簡介
深度學習（Deep Learning）與人工智慧（Artificial Intelligence, AI）近年來已成為各大科技公司投入的核心研究領域。其中，影像識別技術 因具備廣泛的應用價值而備受關注。在深度學習模型中，卷積神經網路（Convolutional Neural Network, CNN） 是影像辨識的主要架構，而卷積運算則為整體運算中最關鍵且最耗時的部分。由於卷積運算涉及大量重複且複雜的資料處理，傳統以 ARM 或 x86 為主的處理器架構 常面臨運算效能瓶頸。相較之下，FPGA（Field Programmable Gate Array） 具備高度並行運算特性，能有效加速此類運算流程，進而顯著提升邊緣運算的整體效能。

---

## 研究背景
#### 深度學習（Deep Learning）與人工智慧（Artificial Intelligence, AI）近年來已成為各大科技公司投入的核心研究領域。其中，影像識別技術 因具備廣泛的應用價值而備受關注。在深度學習模型中，卷積神經網路（Convolutional Neural Network, CNN） 是影像辨識的主要架構，而卷積運算則為整體運算中最關鍵且最耗時的部分。由於卷積運算涉及大量重複且複雜的資料處理，傳統以 ARM 或 x86 為主的處理器架構 常面臨運算效能瓶頸。相較之下，FPGA（Field Programmable Gate Array） 具備高度並行運算特性，能有效加速此類運算流程，進而顯著提升邊緣運算的整體效能。

---
## 實作與開發環境
- **硬體平台**：Xilinx PYNQ-ZU (Zynq UltraScale+)  
- **開發工具**：Vitis HLS、Vivado、PYNQ Framework  
- **開發語言**：C / C++（硬體架構）、Python（控制與驗證）  
---

## 主要系統架構設計
### EPIC（Enhanced Partition for Independent Convolution）架構設計背景 :
在PYNQ-ZU平台（ 基於FPGA而非ARM ）上執行卷積運算時，未採用任何專門的優化策略實現運算，會導致 FPGA 無法正常發揮出並行運算的優勢，造成運算速度低下，無法與利用PyTorch內建的Conv2d運算函數相比。所以需要重新設計卷積的運算架構，使 FPGA 能充分發揮出價值。
起初本論文提出一平行處理的架構方式，雖然證明方向正確，但因為有較大的硬體資源限制問題，運算效能也不達標，所以就進一步提出 EPIC 架構。

### EPIC 架構說明 :
本論文提出一種名為 EPIC 的架構優化方案。該方法透過輸入資料的分割策略，有效解決了卷積運算中需重複載入輸入數據的造成資源不足的問題。EPIC 架構將原先單一的大型卷積模組拆分為 四個可獨立運作的小型卷積模組，使各模組能同時執行運算，並且每個模組內的運算也能同時運算，實現更高層次的平行化。此設計可以充分發揮 FPGA 的高度並行運算能力，大幅強化卷積層的整體執行效能。

- **輸入與權重分割**：
  將輸入與權重資料分割為2個部分，以解決記憶體存取計算的衝突。
- **多運算模組平行化**：  
  使用四個 `Conv` 模組進行並行運算，且4個conv模組內的運算也全部展開進行並行運算處理，大幅提升資料吞吐量。
- **資料流優化 (Dataflow)**：  
  運用 HLS dataflow 技術使資料傳輸與運算重疊，降低延遲。
- **參數彈性化設計**：  
  FPGA具有易修改的特色，可支援不同卷積參數設定，便於應用於不同卷積層可做調整使用。
  
### EPIC 架構
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/a7e12ce5-4ef9-4228-8b46-3ab39c02b87e" />

### Parallel Processing 模組
<img width="500" height="293" alt="圖片200" src="https://github.com/user-attachments/assets/acef31b0-70e0-4153-83b2-d67d63b79ffa" />



### 🔹 各模組功能說明 :
| 模組名稱 | 功能描述 |
|-----------|-----------|
| `EPIC` | 完整的卷積運算的流程 |
| `Parallel Processing` | 處理並行運算的主要模組 |
| `Conv` | 卷積運算的核心計算模組 |
|  `Output add` | 將四個Conv計算出來的結果做相加 |

---
## 硬體資源優化設計
### 1. 卷積IP共用
#### 共用原因 :
IP 也稱 IP core，類似於是硬體中的函式庫，每一個 IP 都可以代表成一個獨立完整的功能模組。而卷積IP共用的核心原因在於，進行卷積運算時會涉及到多層迴圈的處理，而在 FPGA 中進行 for 迴圈的硬體實現時，迴圈變數的範圍（如 p、q、r 等）需要在電路佈局和資源分配之前預先確定。然而，不同卷積的參數組合將直接影響這些變數的取值範圍。為了因應硬體上的限制，除了需要針對單一卷積 IP 資料結構進行優化設計外，也必須透過IP共用策略來節省硬體資源。

#### 共用方式:
在硬體資源限制的條件下，為了增強 FPGA 對不同卷積參數的應用能力，本論文提出了一種優化方法，根據卷積運算過程中涉及到的參數進行大致分類，並從中選擇出現頻率最高，具代表性的幾個參數組合，並以這些參數設定為共用模板，製作出對應的模板 IP，此論文稱此 IP 為 TIP (Template IP)，並在 Vivado 中將多個 TIP 寫到同一 FPGA 中，即可同時使用。同時，在Vitis HLS 中根據EPIC的設計進行延伸，使一個TIP下可以使多個卷積計算共用，從而減少每一種參數都需使用一獨立IP的情形。

#### 共用架構設計 :
為了能計算所有共用同一TIP但不同規模的卷積運算，需在原先的EPIC架構中進一步引入對應其運算規模的 Parallel Processing 模組，並引入Check根據不同的運算規模需求進行調配。本論文根據不同卷積操作的計算量，將其分配到相應的 Parallel Processing 模組中，以避免將運算量較輕的卷積運算用上較大計算量的 Parallel Processing 模組，從而防止規模較小的卷積運算進行多餘的卷積操作，導致增加多餘的計算時間。

<img width="400" height="430" alt="image" src="https://github.com/user-attachments/assets/073ff877-eb9e-4ffc-872c-a5daa9f02dc6" />

### 2. 儲存精度優化:
#### 優化背景及原因 :
在未經優化前，，EPIC中卷積運算的計算精度與 PyTorch框架中所使用的 Conv2d 函數保持一致，所有資料均以 float32 格式進行處理。從結果可看出FPGA內部的BRAM（Block Random Access Memory）資源佔用率高達86.11 %，此使用率稍高於LUT（Look-Up Table）的70.72 %，並遠高於LUTRAM（Look-Up Table Random Access Memory）的 29 %及 FF（Flip-Flop）的 44 %、DSP(Digital Signal Processing)的 55%。可以從此結果得出，目前EPIC架構中BRAM的高占用是限制整體FPGA硬體設計擴展的主要瓶頸。


<img width="400" height="328" alt="image" src="https://github.com/user-attachments/assets/08af7a38-e07c-4046-8b2a-ac75433b045e" />

#### 資源瓶頸分析 :
對 BRAM 的用途進行深入的分析後發現，導致 BRAM 資源使用率偏高的最主要因素，在於以 float32 格式儲存的輸入資料及權種資料。每筆數據佔用 32-bit（4 Bytes）的存儲空間，當儲存卷積神經網路( CNN )中
的高維度特徵圖（feature map）時，BRAM 的資源消耗將會顯著增加，特別是在YOLO中，每個特徵圖的維度尺寸都較為龐大，這使得BRAM的存儲需求會進一步提升，進而導致 BRAM 的空間不足。

#### 資源瓶頸優化方式 :
本論文採用 Vitis HLS 中內建的 ap_fixed 指令（Arbitrary Precision Fixed Point Types）取代原先的float32。本研究未採用 float16 或 int8 精度格式來降低數值精度，主要原因在於 Vitis HLS 本身並不支援 float16 格式。而 int8 雖具備降低資源使用的潛力，但需透過特定的量化與反量化方法以表示小數精度，然而 Vitis HLS 並未提供對此類處理流程的直接支援。
最後經過大量測試與分析，本論文將原本以 float32（32 bit）存儲的所有數據，調整為採用 ap_fixed<18,6> 的精度格式。 配置表示數據總位寬為 18 bit，其中整數部分占 6 bit（含 1 bit 的符號位，用以標識正負），小數部分占 18 bit − 6 bit = 12 bit。實驗結果顯示，在此精度設定下，辨識率僅有輕微衰減，仍能維持原模型的 98% 辨識精度，從而達到數值精度與資源利用的良好平衡。

---

## 效能結果
| 指標項目                | FPGA (PYNQ-ZU) | ARM Cortex-A53 | 提升幅度                                    |
|------------------------|----------------|----------------|---------------------------------------------|
| **執行時間**            | 530.37 ms      | 241.99 ms      | 執行時間降幅達 **54.37%**，效能明顯提升       |
| **運算穩定度（標準差）** | 0.11           | 2.48           | 標準差大幅降低，計算時長穩定度顯著提升         |

> 本架構可維持原模型 **98% 的推論準確率**
> 在資源優化後可將資源利用率提升3倍
 
---

## 研究論文
> 劉哲安，《基於 FPGA 之 YOLOv9 平行化加速架構設計與實作》，  
> 國立台北大學通訊工程學系碩士論文，2025。

- 研究核心：  
  針對 YOLOv9 模型卷積層的高運算量問題，設計適用於 FPGA 的加速架構。
- 技術貢獻：  
  - 結合多語言開發（C/C++ 與 Python），實現軟硬體協同運算。  
  - 在有限硬體資源下，進行多維度優化（運算單元配置、記憶體策略）。  
  - 完成實際部署與效能測試，驗證可行性與穩定性。

---


