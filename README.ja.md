# Open-DNeo

[DNeo](https://arxiv.org/abs/2611.21925) の drafter の重みを、手元の機材で作るための道具と手順です。配信エンジンが記録した行から学習します。論文の著者とは関係のない、独立の非公式プロジェクトです。

**状態（0.1.0、2026-11-19）：計画と下調べだけです。重みはまだ無く、この方法がうまくいくことを示す結果もありません。** 段が進むごとに、道具と数と、効かなかったことを足していきます。

[English](README.md)

## これは何か

DNeo は、投機的デコードのための並列 drafter です。小さな model が 1 回の計算で複数の token を提案し、本体の model がそれを検証します。返事は本体が書いた物と同じになります。論文は、Qwen3-4B・温度 0 で、DFlash-2 より 1 round の token が 19〜27% 多いと報告しています。著者は Qwen3-4B 用の重みと推論の code を公開しましたが、学習の code は公開していません。

このプロジェクトは、別の本体のための DNeo の重みを作る方法を確かめ、道具・手順・測った数を公開します。最初の対象は、[TensorFold](https://github.com/ashhart/TensorFold) で配信する GLM-5.3-Flash（GB10 2 台、DGX Spark 級）です。

## なぜ論文の方法をそのまま使わないか

論文は、本体に書かせた文章（rollout）で学習します。本番は 160 万本（1 本 1,024 token まで）、ablation は 10 万本です。こちらの 2 台組では、本体が書く速さ（decode）は 4 本合計で 52〜73 tok/s、読む速さ（prompt の処理）は 1,420〜1,540 tok/s です。

| 作る物 | token の上限 | 道 | 1 組での時間 |
|---|---|---|---|
| 本番の 160 万本 | 16.4 億 | decode | 260〜365 日 |
| ablation の 10 万本 | 1.02 億 | decode | 16〜23 日 |
| 同じ 1.02 億 token | | prompt の処理 | 18〜20 時間 |

読ませる方が、書かせるより 1 token あたり約 20〜30 倍安く済みます。計画では、どの案も「どちらの値段が要るか」で並べています。

## 進め方

1. **書かせずに、記録する。** 既にある幅広い文章を prompt として本体に読ませ、drafter が読む行を記録します。TensorFold の「drafter の学習」の手順書に、別の本体でこの形が効いたという報告があります。
2. **配布元の DFlash2 の重みから始める。** 一からは学習しません。層と head は同じ並びなので引き継げます。入口は引き継げないので（下調べを参照）、最小二乗で橋を渡し、効きが悪ければ入口の周りだけ初期化し直します。
3. **rejection memory は最後にする。** DNeo の部品のうち、論文が「drafter 自身の提案で回した記録が要る」と書いているのはこれだけです。測られた 4 つの効果の中でも最小です。
4. **エンジンの code を書く前に、手元で測る。** PyTorch の物差しで、記録した行の上で重みを採点します。配信の時と同じ制約をかけます。本体が今持っている 2 つの drafter の両方に勝ってから、エンジン側を書きます。

段取り・通過の条件・止める条件は [docs/plan.md](docs/plan.md)（英語）にあります。

## 下調べで分かったこと

- 公開されている DNeo の重みと、GLM 用の DFlash2 の drafter は、層の tensor の並び（1 層 15 個）、最後の norm、head の 2 つが同じです。DNeo は、密な入口の行列を、本体の層ごとに係数 1 個の形に替えています。
- GLM 用の DFlash2 の入口は、座標を完全に混ぜる行列でした。層ごとの block の対角の割合は 0.024% で、偶然と同じです。後ろの層はその混ざった座標の上で学習されているので、入口はそのままでは移せません。
- TensorFold の GLM の経路では、はねられた提案の行が、drafter の読む verify の buffer に既に残っています。rejection memory が要る入力です。

出どころ・revision・数の全体は [docs/survey.md](docs/survey.md)（英語）、生の出力は [receipts/](receipts/) にあります。

## この版の道具

```sh
python -m pip install -e .
open-dneo shapes incoai/GLM-5.3-Flash-DFlash2     # file の一覧、config、全 tensor の形
open-dneo entrance incoai/GLM-5.3-Flash-DFlash2   # 密な入口の行列の、対角の割合
```

どちらも、Hugging Face の公開リポジトリを HTTP の range で読みます。認証は要りません。`shapes` は safetensors の header だけを読みます。`entrance` は tensor を 1 個だけ取得し（この例では 160 MiB）、数 GB の memory を使います。

## 出どころと license

- DNeo：Sandler ほか、[arXiv:2611.21925](https://arxiv.org/abs/2611.21925)。推論の参照 code は [J-sandler/DNeo](https://github.com/J-sandler/DNeo)（MIT）、重みは [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo)（Apache-2.0）。
- DFlash：[z-lab/dflash](https://github.com/z-lab/dflash)（MIT）。GLM 用の DFlash2 の drafter：[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（Apache-2.0）。
- [TensorFold](https://github.com/ashhart/TensorFold)（Apache-2.0）：エンジン、その recorder、drafter の学習の手順書。

この版には、上の code も重みも含まれていません。Open-DNeo の license は [Apache-2.0](LICENSE) です。DNeo の著者、Cerebras、incoai、Z.ai、TensorFold のメンテナとは関係がありません。
