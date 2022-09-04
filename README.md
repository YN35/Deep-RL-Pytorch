# Deep-RL-Pytorch
deep reinforcement learning pipeline with pytorch

複数のモデルを実装するにあたり使いまわせるコード

## How to use

`hogehoge.yaml`に学習する上での設定項目を書き込む

このフォルダのディレクトリ上で以下のコマンドを実行
```
# 学習
python train.py --config config/train_cls_01.yaml
# 評価
python test.py --config config/test_cls_01.yaml
```
中間・最終結果は runs_* フォルダにダンプされる

## structure
* mllib : 主要なライブラリ
* config : 設定ファイル置かれている
* main_models/hogehoge に実装したいモデルを実装する


## policy
- なるべく設定ファイルで実行される中身を全部コントロールできるようにする
