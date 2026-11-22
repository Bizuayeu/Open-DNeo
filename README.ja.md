# Open-DNeo

[DNeo](https://arxiv.org/abs/2611.21925) の drafter の重みを、手元の機材で作るための道具と手順です。配信エンジンが記録した行から学習します。論文の著者とは関係のない、独立の非公式プロジェクトです。

**状態（0.7.0、2026-11-23）：計画と下調べと、配信と同じに数える手元の物差し、Stage 1 の試しの記録と層の probe と学習の前の橋、それに drafter を学習する command までです。重みはまだ無く、この方法がうまくいくことを示す結果もありません。** 門 G0 で、物差しの「1 round の token」は、評価の組のうち封をしていない半分の、どの class でも配信の数と 0.013 以内でした（[受領証](receipts/2026-11-20-g0/README.md)、英語）。記録の安い prompt の通り道の行からでも、公開されている DFlash2 の drafter は、同じ token の返事の行からとほぼ同じ案を出しました（[受領証](receipts/2026-11-20-rows/README.md)、英語）。試しに記録したその行 609,309 行では、最後の行の手前の 5 つの tap（40〜44）が、DFlash2 の読む 5 つより、本体の次の token を多く持っていました（[受領証](receipts/2026-11-20-pilot/README.md)、英語）。DNeo の入口を最小二乗で DFlash2 の層へつなぐと、drafter は学習の前から DFlash2 の当たった draft の 0.853〜0.962 を当てました。その深い 5 つの tap からの橋は、DFlash2 自身の 5 つからの橋より 5 class のうち 3 つで上でしたが、数の前に置いた規則に届かず、Stage 2 は DFlash2 の tap を読みます（[受領証](receipts/2026-11-21-g1/README.md)、英語）。この版が足すのは学習の command だけで、その最初の走りの数はまだ入っていません。段が進むごとに、道具と数と、効かなかったことを足していきます。

[English](README.md)

## これは何か

DNeo は、投機的デコードのための並列 drafter です。小さな model が 1 回の計算で複数の token を提案し、本体の model がそれを検証します。返事は本体が書いた物と同じになります。論文は、Qwen3-4B・温度 0 で、DFlash-2 より 1 round の token が 19〜27% 多いと報告しています。著者は Qwen3-4B 用の重みと推論の code を公開しましたが、学習の code は公開していません。

このプロジェクトは、別の本体のための DNeo の重みを作る方法を確かめ、道具・手順・測った数を公開します。最初の対象は、[TensorFold](https://github.com/ashhart/TensorFold) で配信する GLM-5.3-Flash（GB10 2 台、DGX Spark 級）です。

## 並列の drafter のしくみ（初めての方へ）

**投機的デコード。** 小さな model（drafter）が次の数 token を推量し、本体の model がそのすべてを 1 回の計算でまとめて確かめます。本体は位置ごとに、一人で書く時と同じやり方で自分の token を選び、推量がそれと同じなら受け取ります。最初に違った位置では、推量を捨てて自分の token を書きます。ですから返事は、drafter を使っても使わなくても変わりません。本体が温度つきで書く時も同じです（TensorFold は sampling の乱数を決まった鍵から引くので、同じ文が出ます。一般的な speculative sampling では、同じ文ではなく同じ分布になります）。この 1 回の確かめを 1 round と呼びます。1 round で受け取られる推量が多いほど、返事に要る本体の計算の回数が減ります。

**drafter は温度 0 で推量する。** 本体が温度つきで token を引く時、1 つの推量が受け取られる見込みは、本体がその token を引く確率です。ですから推量を決め打ちで 1 つ出すなら、本体がいちばん引きそうな token が最もよく当たります。drafter は温度 0（毎回いちばん確率の高い token を選ぶ書き方）で推量し、このプロジェクトの drafter も、本体のいちばんの token を当てるように学ばせます。

**推量の書き方は 2 通り。** 逐次の drafter（EAGLE や MTP の head）は、推量を 1 つずつ書きます。推量 1 つごとに drafter をもう 1 回動かし、その推量は前の推量を見ています。並列の drafter（DFlash や DNeo）は、block 全体を 1 回の計算で書きます。block の行は mask の token から始まるので、どの行も、前の行が何を推量したかを知りません。1 round に 1 回の計算は、推量ごとに 1 回より安く済みます。代わりに、後ろの推量は前の推量を知らないまま書かれ、当たりにくくなります。

**DNeo は、推量を 1 つずつ書く代わりに何をしているか。** drafter の層は、1 round に 1 回だけ動かします。そのうえで、後ろの推量に足りない情報を、安く済む所で補います。

- **前の推量を、head だけが見る（causal な head）。** 位置を 1 つずつ進むのは、head（状態を token の点数に変える最後の小さな段）だけです。head は、前の位置で選んだ token を見て、語彙のすべての点数を直します。DFlash2 の head は、上位 16 個の候補についてこれをします。DNeo の head は、低ランクの補正で、全部の候補について行います。
- **前の round の外れを、次の round で見る（rejection memory）。** 推量がはねられた位置の本体の hidden state を、次の round の drafter の context に足します。検証の時に本体が既に計算した物なので、param も本体の計算も増えません。このプロジェクトの最初の版には入れません（[進め方](#進め方)を参照）。
- **block の中の行どうしは、互いを読まない。** 論文では、読ませない方がわずかに良く、計算も安く済みました。

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
python -m pip install -e ".[torch]"               # ほかの command は torch で計算する
open-dneo shapes incoai/GLM-5.3-Flash-DFlash2     # file の一覧、config、全 tensor の形
open-dneo entrance incoai/GLM-5.3-Flash-DFlash2   # 密な入口の行列の、対角の割合
open-dneo recording DIR                           # server が --record DIR で記録した物の検査
open-dneo greedy DIR --target CHECKPOINT          # 記録のうち、本体が自分で書いたはずの token の数
open-dneo greedy DIR --target CHECKPOINT --save NEXT   # それに加え、行ごとの本体の選びを残す
open-dneo greedy DIR --target CHECKPOINT --save NEXT --top K   # 行ごとの上位 K とその確からしさも残す
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT   # その記録の上での、DFlash2 の 1 round の token
open-dneo rows DIR --drafter DRAFTER --target CHECKPOINT --run RUN   # prompt の行と返事の行からの案の比べ
open-dneo probe DIR --target CHECKPOINT --next NEXT   # どの tap が本体の次の token を持っているか
open-dneo bridge DIR --drafter DRAFTER --out OUT   # DNeo の入口を drafter の層へつなぐ当てはめ
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT --bridge BRIDGE   # その入口で数える
open-dneo score DIR --drafter DRAFTER --target CHECKPOINT --bridge BRIDGE --dneo-mask --dneo-head 0   # DNeo の形で数える
open-dneo train DIR... --next NEXT --drafter DRAFTER --target CHECKPOINT --out OUT --steps N   # drafter の学習
```

`shapes` と `entrance` は、Hugging Face の公開リポジトリを HTTP の range で読みます。認証は要りません。`shapes` は safetensors の header だけを読みます。`entrance` は tensor を 1 個だけ取得し（この例では 160 MiB）、数 GB の memory を使います。

`recording` は、TensorFold の server が `--record` で書いた file を読みます。stream ごとの token・位置・種別（prompt の行か、返事の行か）・tap した行と、logprobs を頼んだ request では本体の上位の token です。閉じた stream を 1 本ずつ recorder の書き方と照らし、行の形ごとに合計を出します。合わない stream があれば、終了 status は 1 です。`--streams` を付けると、stream を 1 本ずつ並べます。Python からは、`open_dneo.recording.open_stream` が stream の行を配列で返します。

`greedy` と `score` は、本物の重みの上で動きます。要る物は、記録と、本体の checkpoint の folder と、`score` では drafter の folder です。tensor の名は GLM-5.3-Flash の物を読みます。head は、そのままの形でも MXFP8 でも読めます。checkpoint の folder には、`config.json`、index、埋め込みと head の入った file があれば足ります。

`greedy` は、最後の norm のあとの行に本体の head を掛け、記録の token のうち「本体が次に書くはずの token」と同じ物を、返事と prompt に分けて数えます。sampling なしで書かせた返事なら、全部が同じになるはずです。drafter を採点する前に、head の tensor・その向き・tap が正しいことを、これで確かめます。`--save NEXT` を付けると、数えた stream の行ごとの本体の選びを、1 行に little-endian の int32 を 1 つずつ `NEXT/<stream>.next` に残します。part ごと・役割ごとの数えと `probe` に使います。`--top K` を足すと、行ごとの上位 K の token とその対数確率を `NEXT/<stream>.best` に、次の token の対数確率を `NEXT/<stream>.nextlp` に残します。

`score` は、返事の行を 1 つずつ anchor にして、配信の時に組まれるのと同じ block を DFlash2 の drafter に通します。文脈は、その stream のそれより前の行です。提案の鎖は TensorFold の GLM の lane と同じ所で切り、返事を 2 通りに歩きます。1 つ目は、どの round も drafter の物として数えます。2 つ目は、1 本だけで書かせた stream の round を、配信が取るとおりに数えます。stream の最後の token が、それより前の 8 token 以上の並びを繰り返している所では、その round は drafter の鎖の代わりに「前にその並びの後へ続いた token」の写し（copy）を検証します。`--served` を付けると、重みを配信の時の既定の形にして採点します。drafter の線形と本体の head は 4 bit、head の点数は bfloat16 です。`--max-tokens` には request の新しい token の上限を渡します。上限で切られた返事の終わり近くの round が、それで決まります。

`--run RUN` は、走りの一覧（run file、`open_dneo.runs`）を読みます。1 行が 1 つの request で、prompt と返事の token id、`max_tokens`、配信の round の数を持ちます。request ごとに、それを記録した stream を探して組み、request の prompt の全部と返事の全部を、その request 自身の上限まで歩きます。server が保存した prompt の頭から再開して stream が prompt の途中から始まっていても、prompt の頭からの copy を見落としません。報告は、配信の「1 round の token」と物差しの数を class ごとに並べます。これに `--drafts-log LOG` を足すと、TensorFold の server が `TF_GLM_DFLASH2_LOG` で書いた DFlash2 の round の行を読み、同じ anchor で最初の draft が物差しと同じ round、頭から同じ draft の数、全部の draft が同じ round を class ごとに数えます。

[計画](docs/plan.md)（英語）の門 G0 が配信と比べるのは、2 つ目の歩みの数です。`--run` を付け、重みを配信の形にして比べると、どの class でも 0.013 以内でした（[receipts/2026-11-20-g0](receipts/2026-11-20-g0/README.md)、英語）。配信との違いは 2 つ残っています。返事の最初の round が持つ draft の数は、配信では実測の round の時間から決まります。報告は、その round が draft を 1 本だけ持った時と、全部持った時の round の数も出します。また、`--run` を付けない時に物差しが歩くのは記録が持つ token で、返事より 1 つ少ないことがあります。

`rows` は、drafter の学習に使える 2 種類の行を比べます。run file の返事を記録し、続けて同じ token を 1 つの prompt として送り直した記録では、返事の位置ごとに、返事の round で計算された行と、prompt の通り道で計算された行の 2 つがあります。`rows` は request ごとに 2 本の stream を組み、両方を「最初の行の遅い方」から読み、返事の位置ごとに、それぞれの行から drafter に案を出させます。報告は class ごとに、それぞれの行での drafter だけの歩みと、2 つの案が最初の draft で、また全部の draft で同じだった割合を出します。重みを配信の形にすると、歩みの差は 1 round あたり 0.004〜0.029 token でした（[receipts/2026-11-20-rows](receipts/2026-11-20-rows/README.md)、英語）。

`probe` は、記録のどの tap が「本体が次に書く物」を持っているかを調べます。tap ごとに、また並べた tap の組（`--set 6,15,25,34,43`）ごとに、位置 p の行から位置 p + k の本体の最後の行（`--k`、既定は 1・2・3）への切片つきの線形の写しを、大半の stream の上で ridge の最小二乗で当てはめます。名の順に `--held-every` 本に 1 本を取っておき、その stream で、写した行を本体の head に通し、一番の token が p + k の本体自身の選び（`greedy --save` が残した物）と同じ割合を数えます。床（一番多い選びがそのまま当たる割合）も出し、`--parts`（1 行に stream の名とその part）があれば part ごとにも出します。最後の tap の k 0 は行をそれ自身へ写すので 1 になり、選びが行と並んでいることの確かめになります。

`bridge` は、DNeo の入口を DFlash2 の drafter の層へつなぎます。DNeo の入口は tap の行の重み付き和で、DFlash2 の入口は 5 つの tap の行を並べて `fc` で混ぜます。`bridge` は、与えた folder の行の上で、重み付き和から `fc` の出力への行列 B を ridge の最小二乗で当てはめます（どちらも `fc` の出力の大きさで割ります）。名の順に `--held-every` 本に 1 本を取っておき、その行で `fc` の出力を説明する割合と cos を出し、`--parts` があれば part ごとにも出します。書くのは `bridge-LABEL.safetensors`（B と重みと tap）で、一様の重み、tap の大きさをそろえる重み、`--weights` で与えた重みの分です。`--inputs` で、和が drafter と別の tap を読めます（目標は drafter 自身の tap の上の `fc` のまま）。`score --bridge FILE` は、その入口で drafter を数えます。入口は DNeo の重み付き和、B、DFlash2 の norm で、層・mask・head は DFlash2 のままです。B は float32 のままで、`--bridge-served` を付けると他と一緒に 4 bit に丸めます。`--run` を付けると、`score` は class ごとに、copy なしで毎 round を drafter が出す歩みも出します。学習の前（step 0）の橋つきの drafter は、DFlash2 の当たった draft の 0.853〜0.962 を当てました（[receipts/2026-11-21-g1](receipts/2026-11-21-g1/README.md)、英語）。

`score --dneo-mask` は DNeo の mask（block の行が見るのは文脈と、その block の最初の行だけ）で、`--dneo-head K` は DFlash2 の head から引き継いだ DNeo の head（語彙の全部なら 0、本体の head の上位 K なら K）で数えます。K を DFlash2 自身の値にすると、DFlash2 と同じに draft します。`--run` を付けると、報告は class ごとに次も出します。前の draft が当たった後の位置ごとの当たり（draft は全部持つ）、配信の cut が読む確からしさの積に対する鎖ごとの当たり、cut を 0.2〜0.9 にした時の 1 round で検証する token と draft、確からしさに対する当たりの割合、drafter の状態の上の本体の head の上位 k が次の token を持つ割合です。

`train` は、記録した行で drafter を学習します。始めは DFlash2 の drafter（`--bridge`・`--dneo-mask`・`--dneo-head` は `score` と同じ）か、学習した drafter です。正解（label）は `greedy --save` が残した本体自身の選び（`--next`）で、anchor の block に、その後の行の本体の選びを cross-entropy で教えます。教えるのは本体の書く行（GLM の chat の型の assistant の turn か、型の無い平文）の間だけです。借りた文章では、本体が選ばなかったはずの最初の token より後の label は、本体の書かなかった文に続くので、`--off-path` がその重みを決めます（既定は 0）。draft の重みは位置とともに `--decay` で下がります。1 step は、part ごとに 1 本の stream から `--anchors` 個の anchor の窓を取り（`--parts` は 1 行に stream の名とその part）、part の損失を `--shares` の割合で足し、AdamW で進みます。率は `--warmup` step で `--peak` まで上げ、cosine で `--end` まで下げ、drafter の部分ごとの係数（`--rates`）を掛け、勾配は切り詰めます。名の順に `--held-every` 本に 1 本を取っておき、`--measure-every` step ごとにその窓で、part の全部と part ごとに drafter を測ります。`--keep-every` step ごとに `--out` の下へ `step-N` の folder を残します。`score` はその folder を、学習した時の入口・mask・head で読み、`--resume` はそこから続けます。`--device cuda` で GPU の上で学習します。既定の値はどれも仮置きで、出所は `trainer.Settings` にあります。

これらの command が要る記録：tap k は「本体の k 層を通ったあとの状態」です。GLM では 4 本の residual stream の平均で、DFlash2 の drafter が読む物と同じです。層の数と同じ番号の tap は、最後の norm のあとの行です。`greedy` と `probe` はその最後の tap（GLM-5.3-Flash では 45）を読みます。`score`・`rows`・`train` は、drafter の `target_layer_ids` のそれぞれに 1 を足した tap（上の drafter では 6・15・25・34・43）を読み、`--bridge` があれば橋の名指す tap を読みます。TensorFold の `--record` は、GLM の行をまだ出しません。このプロジェクトが使っている GLM 用の recorder も、まだ公開していません。その最初の確かめで分かった事は、[receipts/2026-11-19-recorder-check](receipts/2026-11-19-recorder-check/README.md)（英語）にあります。

## 出どころと license

- DNeo：Sandler ほか、[arXiv:2611.21925](https://arxiv.org/abs/2611.21925)。推論の参照 code は [J-sandler/DNeo](https://github.com/J-sandler/DNeo)（MIT）、重みは [J-sandler/Qwen3-4B-DNeo](https://huggingface.co/J-sandler/Qwen3-4B-DNeo)（Apache-2.0）。
- DFlash：[z-lab/dflash](https://github.com/z-lab/dflash)（MIT）。GLM 用の DFlash2 の drafter：[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（Apache-2.0）。
- [TensorFold](https://github.com/ashhart/TensorFold)（Apache-2.0）：エンジン、その recorder、drafter の学習の手順書。

この版には、上の重みは含まれておらず、file の写しもありません。一部の module は、上の code と同じ計算をするように書きました。どの module がどれに沿っているかは [NOTICE](NOTICE) にあります。Open-DNeo の license は [Apache-2.0](LICENSE) です。DNeo の著者、Cerebras、incoai、Z.ai、TensorFold のメンテナとは関係がありません。
