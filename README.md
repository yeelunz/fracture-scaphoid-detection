# 材料及方法

## 資料集的準備

### 舟骨辨識部份

舟骨辨識部分採用分割訓練集與測試集8:2，其中訓練集共159張，測試集為40張。

資料擴增採用水平翻轉、隨機旋轉90度、隨機縮放。

### 骨折辨識部份

而骨折部分則用舟骨部份的原始圖片(測試集與訓練集部份與舟骨一樣)，並裁切舟骨區域的ground truth進行訓練，其中有骨折的圖片佔了49.38%，未骨折佔了50.62%。

資料擴增採用水平翻轉、隨機縮放，排除掉部份資料的骨折的bounding box若超出舟骨標記的範圍，則該筆不進行資料擴增。

## 方法

### 第一階段(舟骨辨識)

舟骨區域辨識採用pytorch內的預訓練的fasterrcnn_resnet50_fpn_v2，optimize採用AdamW。

### 第二階段(骨折辨識)

這邊backbone採用預訓練的resnet101，加入FPN與SpatialAttention，並且訓練他回歸$(x,y,h,w,a)$這五個參數，其中$x,y$ 代表預測bounding box中心的座標，$h,w$ 代表bounding box的長及寬，$a$ 則代表bounding box旋轉的角度，並且限制在$[\pi/2,-\pi/2]$。

由於模型的IoU與Accrucy/precision/recall難以同時達到高的性能(但還是有訓練出一個平衡IoU效能與Accrucy的模型)。

於是選擇同時train 兩種類型的模型，IoU效果好的當作偵測IoU，分類效果好的就讓他來做辨識。

不過這樣做bounding box的效果或許就不如真的實際分開做成兩個模型。

之後發現IoU的效能不佳(卡在0.32~0.35)，思考有可能是在$[\pi/2,-\pi/2]$ 的回歸方法問題會造成不連續(當角度分別靠近$\pi/2,-\pi/2$ 時，真實相差的角度很小，但是造成的懲罰卻差很多)

於是讓他改成回歸$(x,y,h,w,\cos,\sin)$這六個參數，但這個新模型的效能在辨識的效能上表現大幅的落後前一個模型(辨識部分會降到幾乎不可用的程度)，但在回歸框的表現略優於前一個五個參數的版本的。但優勢實在不明顯(IoU約36-37%)，因此還是沿用原本的5個參數的模型。

---

# 結果與討論

### 第一階段(舟骨辨識)

第一階段在使用Faster Rcnn Fpn v2的情況下，對於測試集來說，在大部分的情況都可以抓出舟骨，辨識舟骨的成功率約在92%左右。

並且所圈出的bounding box都與Ground Truth 相差不大，因此第一階段可以看做成果相對成功。

### 第二階段(骨折辨識)

這裡發現訓練模型的時候，在嘗試提高IoU的表現的話往往會造成Accrucy, Precision, Recall的下降。

因為模型為了提高IOU以及坐標的回歸表現會傾向於能畫框就盡量畫框(也可能是我的loss fuction設計不佳)

因此第二階段的骨折辨識以及bounding box的位置是分成兩個模型來檢測。

以下我一共訓練了4個模型，分別是：

- ResNet152_Detction ：使用ResNet152作為backbone 的專門辨識骨折有無的模型
- ResNet101_Detction：使用ResNet101作為backbone 的專門辨識骨折的模型
- ResNet101_IoU：使用ResNet101作為backbone 的專門檢測bounding box 位置的模型
- ResNet101_Balanced：使用ResNet101作為backbone ，平衡IoU以及辨識表現的模型

另外還測試了混和ResNet152_Detction, ResNet101_Detction, ResNet101_Balanced來進行骨折確認，當有兩個認為此圖有骨折，則認為有骨折。

|  | ResNet152_Det | ResNet101_Balanced | ResNet101_IoU | ResNet101_Det | Mixed |
| --- | --- | --- | --- | --- | --- |
| Accuracy (%) | 90% | 92% | 75% | **95%** | **95%** |
| Precision (%) | 83% | 88% | 66% | **93%** | 88% |
| Recall (%) | 93% | 93% | 75% | 93% | **100%** |
| IoU (%) | 30% | 35% | **36%** | 31% | x |

`註` 這邊的測試集是採用從舟骨原始圖片的GroundTruth切割下來的舟骨部分的影像

並且訓練出的模型會頃向於：框出比GroundTruth還大的區域來提升IoU的表現，而角度往往會預測錯誤，就像是會去把旋轉框當作非旋轉的預測以包住GroundTruth，而這與預期想的想要訓練達到的目標並不一樣。

因此，若預測的重點是放在「角度」的正確性來說的話，我認為，直接預測四個頂點的座標會更加的好，而如果是回歸$(x,y,h,w,a)$五個參數的話，角度對於整體的相較於其他的部分影響太小了，前面四個參數若能正確預測，則整體表現也不至於太差，導致了模型會更加頃向於去預測出Ground Truth的外接框以保證IoU的表現不至於太差。

但如果是這樣的話，那直接用Faster Rcnn來預測骨折區的外接框似乎也能達到跟現在這個模型相似的表現。

![IoU = 48%](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2aa755b-2a79-4338-a5a0-ad1e060a7ae1/942aebf4-7b15-465d-a1ca-1a9f6a98efab/image.png)

IoU = 48%

![IoU = 57%](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2aa755b-2a79-4338-a5a0-ad1e060a7ae1/509ce384-d41e-4f40-aabd-f0567873cab7/image.png)

IoU = 57%

### 結合階段

第一階段在大部分的時候辨識順利，在少數圖片第一階段就抓不到舟骨的位置。在抓到舟骨的時候，第二階段的骨折部分的探測模型表現不俗。而在實際預測迴歸框的表現，雖然IoU表現不差，但實際的觀感，例如左圖觀感明顯會不如右圖，但IoU計算出來的結果反而是左圖高

![IoU = 44%](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2aa755b-2a79-4338-a5a0-ad1e060a7ae1/9f77c310-cf07-40d1-8131-b3a0adfc29d5/image.png)

IoU = 44%

![IoU = 42%](https://prod-files-secure.s3.us-west-2.amazonaws.com/f2aa755b-2a79-4338-a5a0-ad1e060a7ae1/ba316a75-6da8-4a83-bb5c-cd5757ce2896/image.png)

IoU = 42%

另外，這個模型在骨折區域佔舟骨的區域很大的情況的話(對大目標的話)，預測效果往往來的比小目標好得多。
