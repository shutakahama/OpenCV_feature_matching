# OpenCV_feature_matching

<img src="https://github.com/shutakahama/OpenCV_feature_matching/blob/master/result/sushi.jpg" width="500">

Hackday2019「都会のオアスシ」で使用した魚へんマップの特徴量マッチングです．  
Python OpenCVでAKAZE特徴量を用いたマッチングを行なっています．  
カメラ画像を入力として，マップ上での現在位置と方向を出力します．  
詳細はこちらの記事を参照してください．  

[OpenCVで他のどの記事よりも頑強に特徴量マッチングしてみた](https://qiita.com/grouse324/items/74988134a9073568b32d)

## 実行環境
```
python = 3.7
opencv-python = 4.1
numpy = 1.17
matplotlib = 3.1
pillow = 6.2
```

## 実行
以下を実行すれば予測値と画像が表示される． 
```
python pattern_matching.py
```

カメラ画像ファイルは[img/query/](https://github.com/shutakahama/OpenCV_feature_matching/tree/master/img/query)にあるので適宜変更可能

## 実行結果例
左が入力カメラ画像，右がマップ画像．  
画像中の線はAKAZEによるマッチングの結果．  
マップ上の青丸が予測位置，赤矢印がカメラの上が向いている方向．  
正しく予測できているのがわかる．  

<img src="https://github.com/shutakahama/OpenCV_feature_matching/blob/master/result/result.png" width="500">
