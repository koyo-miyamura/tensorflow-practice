## README
入力した手書き画像(28×28)が何の文字であるか判定するスクリプト

モデルはTensorFlowのチュートリアルを参考に作成(MNIST)
https://www.tensorflow.org/tutorials/

## Quick start
`python main.py images/three.png`

* 1回目はモデルを生成し、2回目以降は作成済みモデルがあればそれを読み込む
* 入力画像は1つまで
* 画像データは28×28(背景が白で文字色が黒)

## How to generate input data
手書きの画像データの作成は以下を参照してペイント(Windows)で行った
https://www.sejuku.net/blog/44411