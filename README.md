# Weather Classifier on KV260
有鑑於搞這個專題實在是有太多名詞使人霧煞煞，本文最後有稍作整理供參


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
*補圖片
* Ubuntu
右上角設定>網路
*補圖片
* Mac

## 專案架構
1. 訓練模型(電腦)

2. 模型優化(Ubuntu工作站)
模型優化須經以下步驟
  * (檢查 Inspect)(非必要但建議做一下)
    Vitis-AI官網建議你各位將模型進行優化之前要先檢查(有個notebook有飯粒)
    檢查完之後會產生一些檔案，主要包含一個新的.py檔，跟一個.txt文件(裡面有模型調整建議)。
  * Quantize
  * Compile
3. 導入KV260
  * DPU Overlay
  
## 名詞大補帖
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
  
