# Weather Classifier on KV260

## 使用方式
將kv260_part資料夾放入KV260的Jupyter Notebook中，並直接執行notebook即可進行即時天氣分類。

### 如何與KV260連線
將KV260透過乙太網路線與電腦連線，並將DHCP設定手動(詳細步驟如下)
```
網路位址(IPv4)        10.42.0.2
子網路遮罩(Mask)      255.255.255.0
閘道(Gateway)         10.42.0.1
```
#### Windows
開啟設定>網路和網際網路(可能在左邊選單中)>有線網路>將DHCP改為手動
*補圖片
#### Ubuntu
右上角設定>網路
*補圖片
#### Mac
