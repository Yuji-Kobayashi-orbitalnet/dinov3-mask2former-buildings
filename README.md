# DINOv3とMask2Formerによる建物セグメンテーション

DINOv3 ViT-B/16を凍結した特徴抽出器として使い、特徴変換部、RGB補助CNN、Transformers版Mask2Formerを学習する実装です。
WHU Building DatasetとWaterloo Building Datasetの512×512・1024×1024画像を原寸のまま扱います。

## ダウンロードと手順

- [記事対応版 v0.1.0 の実装ZIP](https://github.com/Yuji-Kobayashi-orbitalnet/dinov3-mask2former-buildings/releases/download/v0.1.0/dinov3_mask2former_article_code.zip)
- [環境構築から学習開始までの手順](ARTICLE.md)
- [リリース一覧](https://github.com/Yuji-Kobayashi-orbitalnet/dinov3-mask2former-buildings/releases)

ZIPには実装、環境定義、設定例、説明文とチェックサムが含まれます。
データセット、DINOv3ソース、事前学習済み重みは別途取得してください。

Linuxサーバーで `mkdir -p ~/zip` を実行し、ダウンロードしたZIPを `~/zip/` に転送します。その後、次のように展開します。

```bash
cd ~
unzip -n ~/zip/dinov3_mask2former_article_code.zip
cd ~/dinov3_mask2former
sha256sum -c SHA256SUMS.article
conda env create -f environment.yml
conda activate dino-whu-m2f
unset PIP_CONSTRAINT
python install_runtime.py
```

続いて [ARTICLE.md](ARTICLE.md) のDINOv3ソース・重みの取得、環境確認、データ準備を実施します。
GitHubが自動生成する「Source code (zip)」は展開先の名前が異なるため、記事の手順では上記の実装ZIPを使用してください。

## 対象環境

| 項目 | 構成 |
|---|---|
| OS | Linux |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Python | 3.11 |
| PyTorch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| Transformers | 4.57.1 |
| DINOv3 | ViT-B/16、全層凍結 |

## 主なファイル

| ファイル | 用途 |
|---|---|
| `environment.yml` / `requirements.txt` | 学習環境の定義 |
| `install_runtime.py` | CUDA対応PyTorchと依存ライブラリの導入 |
| `verify_env.py` | GPUと各コンポーネントの動作確認 |
| `joint_native/prepare_joint.py` | データの構成確認とCOCO索引の作成 |
| `joint_native/waterloo_masks_to_coco.py` | 二値マスクを連結成分に分け、RLE注釈へ変換 |
| `joint_native/train_whu.py` | モデル、データローダー、学習・評価関数 |
| `joint_native/create_initial_checkpoint.py` | 新規学習用の初期値作成 |
| `joint_native/train_to_convergence.py` | 検証損失を監視する学習の入口 |
| `joint_native/sources_input.json` | データの配置例。使用環境に合わせて編集 |

## 学習の運用

学習中は訓練損失とデータセット別の検証損失を確認します。
検証損失の停滞条件を満たした後に、検証損失が最小だった `best_loss.pt` を読み込み、詳細評価を一度実行します。
エポック上限に達しても停止条件を満たしていなければ、詳細評価を実行せず、上限を増やして再開します。

学習開始には `train_to_convergence.py` を使用してください。
`train_whu.py` を直接起動すると、旧方式のエポックごとのAP評価が実行されます。

## データと確認範囲

二値教師マスクの8近傍連結成分を建物インスタンスとして扱います。接触している建物は同じ成分になるため、独立した建物ポリゴンの注釈と完全に一致するとは限りません。

実DINOv3とRTX PRO 6000上で確認済みなのは従来の結合モデルでの学習です。収束制御と初期値作成の追加機能は、ローカルの合成入力とDINO代役を用いて確認しています。本リリースは収束済みモデルや最終精度を提供するものではありません。

第三者のモデル、重み、データセットは各配布元から取得し、それぞれの利用条件を確認してください。

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Transformers Mask2Former](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/mask2former)
- [WHU Building Dataset](https://gpcv.whu.edu.cn/data/building_dataset.html)
- [Waterloo Building Dataset](https://doi.org/10.7910/DVN/EXRA2V)
