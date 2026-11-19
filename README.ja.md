# Open-DNeo

[DNeo](https://arxiv.org/abs/2611.21925) の drafter の重みを、手元の機材で作るための道具と手順です。配信エンジンが記録した行から学習します。論文の著者とは関係のない、独立の非公式プロジェクトです。

**状態（0.2.0、2026-11-19）：計画と下調べと、手元で採点する物差しまでです。重みはまだ無く、この方法がうまくいくことを示す結果もありません。** 物差しは本物の重みの上で動きました。物差しが出す「1 round の token」の数は、配信と比べる門を読むまで公開しません。段が進むごとに、道具と数と、効かなかったことを足していきます。

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
python -m pip install -e .                        # shapes・entrance・recording は numpy だけで動く
python -m pip install -e ".[torch]"               # greedy と score は torch で計算する
open-dneo shapes incoai/GLM-5.3-Flash-DFlash2     # file の一覧、config、全 tensor の形
open-dneo entrance incoai/GLM-5.3-Flash-DFlash2   # 密な入口の行列の、対角の割合
open-dneo recording DIR                           # server が --record DIR で記録した物の検査
open-dneo greedy DIR --target CHECKPOINT          # 記録のうち、本体が自分で書いたはずの token の数
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT   # その記録の上での、DFlash2 の 1 round の token
```

`shapes` と `entrance` は、Hugging Face の公開リポジトリを HTTP の range で読みます。認証は要りません。`shapes` は safetensors の header だけを読みます。`entrance` は tensor を 1 個だけ取得し（この例では 160 MiB）、数 GB の memory を使います。

`recording` は、TensorFold の server が `--record` で書いた file を読みます。stream ごとの token・位置・種別（prompt の行か、返事の行か）・tap した行と、logprobs を頼んだ request では本体の上位の token です。閉じた stream を 1 本ずつ recorder の書き方と照らし、行の形ごとに合計を出します。合わない stream があれば、終了 status は 1 です。`--streams` を付けると、stream を 1 本ずつ並べます。Python からは、`open_dneo.recording.open_stream` が stream の行を配列で返します。

`greedy` と `score` は、本物の重みの上で動きます。要る物は、記録と、本体の checkpoint の folder と、`score` では drafter の folder です。tensor の名は GLM-5.3-Flash の物を読みます。head は、そのままの形でも MXFP8 でも読めます。checkpoint の folder には、`config.json`、index、埋め込みと head の入った file があれば足ります。

`greedy` は、最後の norm のあとの行に本体の head を掛け、記録の token のうち「本体が次に書くはずの token」と同じ物を、返事と prompt に分けて数えます。sampling なしで書かせた返事なら、全部が同じになるはずです。drafter を採点する前に、head の tensor・その向き・tap が正しいことを、これで確かめます。

`score` は、返事の行を 1 つずつ anchor にして、配信の時に組まれるのと同じ block を DFlash2 の drafter に通します。文脈は、その stream のそれより前の行です。提案の鎖は TensorFold の GLM の lane と同じ所で切り、返事を 2 通りに歩きます。1 つ目は、どの round も drafter の物として数えます。2 つ目は、1 本だけで書かせた stream の round を、配信が取るとおりに数えます。stream の最後の token が、それより前の 8 token 以上の並びを繰り返している所では、その round は drafter の鎖の代わりに「前にその並びの後へ続いた token」の写し（copy）を検証します。`--served` を付けると、重みを配信の時の既定の形にして採点します。drafter の線形と本体の head は 4 bit、head の点数は bfloat16 です。`--max-tokens` には request の新しい token の上限を渡します。上限で切られた返事の終わり近くの round が、それで決まります。

[計画](docs/plan.md)（英語）の門 G0 が配信と比べるのは、2 つ目の歩みの数です。まだ比べていないので、その門を読むまで、物差しの「1 round の token」の数は公開しません。配信との違いは 2 つ残っています。返事の最初の round が持つ draft の数は、配信では実測の round の時間から決まります。報告は、その round が draft を 1 本だけ持った時と、全部持った時の round の数も出します。また、物差しが歩くのは記録が持つ token で、返事より 1 つ少ないことがあります。

この 2 つの command が要る記録：tap k は「本体の k 層を通ったあとの状態」です。GLM では 4 本の residual stream の平均で、DFlash2 の drafter が読む物と同じです。層の数と同じ番号の tap は、最後の norm のあとの行です。`greedy` はその最後の tap（GLM-5.3-Flash では 45）を読みます。`score` は、drafter の `target_layer_ids` のそれぞれに 1 を足した tap（上の drafter では 6・15・25・34・43）を読みます。TensorFold の `--record` は、GLM の行をまだ出しません。このプロジェクトが使っている GLM 用の recorder も、まだ公開していません。その最初の確かめで分かった事は、[receipts/2026-11-19-recorder-check](receipts/2026-11-19-recorder-check/README.md)（英語）にあります。

## 出どころと license

- DNeo：Sandler ほか、[arXiv:2611.21925](https://arxiv.org/abs/2611.21925)。推論の参照 code は [J-sandler/DNeo](https://github.com/J-sandler/DNeo)（MIT）、重みは [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo)（Apache-2.0）。
- DFlash：[z-lab/dflash](https://github.com/z-lab/dflash)（MIT）。GLM 用の DFlash2 の drafter：[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（Apache-2.0）。
- [TensorFold](https://github.com/ashhart/TensorFold)（Apache-2.0）：エンジン、その recorder、drafter の学習の手順書。

この版には、上の重みは含まれておらず、file の写しもありません。一部の module は、上の code と同じ計算をするように書きました。どの module がどれに沿っているかは [NOTICE](NOTICE) にあります。Open-DNeo の license は [Apache-2.0](LICENSE) です。DNeo の著者、Cerebras、incoai、Z.ai、TensorFold のメンテナとは関係がありません。
