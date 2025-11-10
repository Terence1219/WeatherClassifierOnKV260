# Weather Classifier on KV260

Since this project involves many technical terms, a glossary is provided at the end for reference.

## Usage

Connect to the KV260 as follows. Place the `kv260_part` folder into the KV260’s Jupyter Notebook environment, and directly run the notebook to perform real-time weather classification.

### How to Connect to KV260

Connect the KV260 to your computer via Ethernet, set the DHCP configuration manually (details below), and save the settings:

```
IP Address (IPv4) 10.42.0.2 
Mask 255.255.255.0 
Gateway 10.42.0.1
```

Open a browser on your computer and enter:

```
10.42.0.3:9090
```

This will connect you to Jupyter Notebook. For device passwords, please contact the teaching assistant.

Common locations for changing network settings:

- **Windows**  
  Settings > Network & Internet (may be in the left menu) > Wired Network > Set DHCP to Manual  
- **Ubuntu**  
  Top-right corner Settings > Network  

## Project Structure

1. **Train the model (on PC)**  
2. **Model optimization (on Ubuntu workstation)**  
   Steps required for optimization:  
   - **Inspect (optional but recommended)**  
     Vitis-AI official documentation suggests inspecting the model before optimization (a sample notebook is provided).  
     After inspection, several files are generated, mainly a new `.py` file and a `.txt` file containing model adjustment suggestions.  
   - **Quantize**  
   - **Compile**  
3. **Deploy to KV260**  
   - **DPU Overlay**  

## Glossary of Terms

### Architecture (Hardware/Software)

- **Field Programmable Gate Array (FPGA)**  
- **KV260**  
  Essentially a small computer with a built-in FPGA. To use the FPGA portion, additional tools are needed to enable communication (such as the Overlay mentioned below).  
- **PYNQ**  
- **Processing System (PS)**  
  Equivalent to the CPU portion of a computer.  
- **Programmable Logic (PL)**  
  Equivalent to the FPGA portion.  
- **Overlay**  
  A bridge connecting PS and PL, allowing them to communicate.  
- **Deep Learning Processor Unit (DPU)**  
  A block implemented on FPGA specifically designed to run deep learning tasks — essentially hardware tailored for deep learning.  

### Model Optimization

- **Inspector**  
- **Quantize**  
- **Compile**

# Weather Classifier on KV260
有鑑於搞這個專題實在是有太多名詞，本文最後有稍作整理供參


## 使用方式
按以下方式與KV260連線。將kv260_part資料夾放入KV260的Jupyter Notebook中，並直接執行notebook即可進行即時天氣分類。

### 如何與KV260連線
將KV260透過乙太網路線與電腦連線，將DHCP設定手動(詳如下)並儲存。
```
網路位址(IPv4)        10.42.0.2
子網路遮罩(Mask)      255.255.255.0
閘道(Gateway)         10.42.0.1
```
開啟電腦瀏覽器並輸入: 
```
10.42.0.3:9090
```
即可連線至Jupyter Notebook，各設備之密碼請洽助教。
下為常見作業系統更改網路設定的位置:
* Windows
開啟設定>網路和網際網路(可能在左邊選單中)>有線網路>將DHCP改為手動
* Ubuntu
右上角設定>網路

## 專案架構
1. 訓練模型(電腦)

2. 模型優化(Ubuntu工作站)
模型優化須經以下步驟
  * (檢查 Inspect)(非必要但建議做一下)
    Vitis-AI官網建議將模型進行優化之前要先檢查(有個notebook有範例)
    檢查完之後會產生一些檔案，主要包含一個新的.py檔，跟一個.txt文件(裡面有模型調整建議)。
  * Quantize
  * Compile
3. 導入KV260
  * DPU Overlay
  
## 名詞大補帖
### 架構相關(軟硬體)
  #### Field Programmable Gate Array (FPGA)
  
  #### KV260
  基本上就是一台內建FPGA的小電腦，但是若要使用FPGA的部分，我們需要一些東西來輔助兩者溝通(例如等等會提到的Overlay)。
  #### PYNQ
  
  #### Processing System (PS)
  相當於電腦CPU的部分
  #### Progarmmable Logic (PL)
  相當於FPGA的部分
  #### Overlay
  簡單講就是連接PS跟PL的橋樑，讓兩者可以互相溝通。
  #### Deep Learning Processor Unit (DPU)
  用FPGA做出一個可以專們拿來跑深度學習相關的Block出來，這個Block就叫做DPU。(幫深度學習量身訂做硬體的部分)  
### 模型優化相關
  #### Inspector
  #### Quantize
  #### Compile
