Qwen-Drive-1.0-4B による ROAD 時系列シーンラベリングの追加評価

作成日 2026-10-20
対象記事 https://zenn.dev/riita10069/articles/cosmos3-driving-video-scene-labeling
論文 DOI 10.5281/zenodo.21863163


1. 結論

Qwen-Drive-1.0-4B は、Cosmos 3 と同一の Evaluation 875 clips、同一の入力動画、同一の Prompt で、Cosmos 3 Super と同等の全体 F1 を示した。
Uniform 24 frames で F1 74.7%、Cosmos 3 Super Adaptive 24 frames は 74.5% であり、差の 95% 信頼区間は -1.1 から +1.6 points で 0 を跨ぐ。
Cosmos 3 Nano Uniform に対しては +5.3 points で、95% 信頼区間は +3.9 から +6.8 points だった。

一方、記事で性能が悪かった 3 タグ（車線変更、停止車両、歩行者横断待ち）は、Cosmos と同じ 13 タグ一括判定のままでは改善しなかった。
車線変更 0.0%、停止車両 37.7%、歩行者横断待ちは 16.1% で、3 タグとも Cosmos 3 Super を下回った。
Qwen-Drive は positive を出す回数が少なく、Precision は高いが Recall が大きく落ちる。

この 3 タグだけを 1 タグずつ別 request で判定し直すと、改善が確認できた。
停止車両は 53.9%、歩行者横断待ちは 43.8% になり、Cosmos 3 Super に対する差の 95% 信頼区間はそれぞれ +1.3 から +15.1 points、+0.1 から +27.2 points で 0 を上回った。
車線変更は 15.4% で Super の 15.0% と同水準であり、改善は確認できなかった。
この条件では全体 Precision が 90.0% から 79.7% へ下がるため、全体 F1 は 74.3% で Super と同等のままである。

記事で採用した固定 VO prior を Qwen-Drive の出力へ再チューニングなしで適用すると、全体 F1 は 77.3% になった。
Cosmos 3 Super に同じ VO prior を適用した 76.2% との差の 95% 信頼区間は -0.2 から +2.5 points で、明確な優位とは言えない。

推論費用は g5.2xlarge で 1,000 clips 当たり USD 1.49、走行動画 1 時間当たり USD 0.67 だった。
Cosmos 3 Super の USD 6.07 の約 9 分の 1、Cosmos 3 Nano の USD 0.31 の約 2.2 倍である。
GPU が記事の実験と異なるため、費用は参考値として扱う必要がある（6 章）。

Qwen-Drive の思考モードを有効にすると、弱いタグの Recall は上がったが、全体では悪化した。
875 clips 中 84 clips で推論過程が 4,096 tokens の上限に達して答えを出せず、coverage が 90.5% に落ちたためである。
strict F1 は 72.6% で、思考モードなしの 74.7% より 2.1 points 低い（95% 信頼区間 -3.5 から -0.7 points）。


2. 何を固定し、何を変えたか

比較の前提として、モデル以外の条件を記事の実験と一致させた。

評価データは記事と同じ ROAD fold 3 の Evaluation 875 clips、11,314 known clip-tag pairs である。
Ground Truth ファイルは記事で使った road-ground-truth.jsonl をそのまま使い、SHA-256 は 17e00dd8 で始まる値で一致する。
ROAD の元動画 15 本は公式配布から取り直し、15 本すべてが記事の dataset lock の SHA-256 と一致した。

入力動画は記事のコードで作り直した。
公開リポジトリから消えていた benchmark コードは、Codex のセッションログに残っていた 584 件の patch を時系列に再適用して復元した。
復元コードで作った request は、Cosmos 3 Nano と Super の Full Run の request_sha256 と 875 件すべてで一致した。
作り直した Uniform 24 frames と Adaptive 24 frames の mp4 も、記事の media lock の SHA-256 と 1,750 件すべてで一致した。
したがって、Cosmos 3 が受け取った動画と Qwen-Drive が受け取った動画はバイト単位で同一である。

System Prompt、13 タグの定義、timestamp manifest、Reasoned Prompt、max_tokens 64、temperature 0、seed、parse 失敗時の repair request も記事と同一にした。
Qwen-Drive の chat template は既定で思考モードに入るため、通常条件では enable_thinking を false にした。

評価指標は記事と同じ定義で、unknown pair を除外し、parse 不能な出力は abstention として strict 指標に残した。
信頼区間は source video 単位の block bootstrap で求めた。
評価コードは記事の Cosmos 3 の 3 run を再評価して、F1、Precision、Recall、Overall Accuracy、Balanced Accuracy、MCC がすべて記事の値と一致することを確認してから使った。


3. EKS 上の推論構成

推論は記事と同じ EKS クラスタ cosmos3-vllm-poc（us-west-2）で行った。
Qwen-Drive 用に GPU node group を追加し、vLLM の Deployment と ClusterIP Service を置き、kubectl port-forward 経由で benchmark client から呼んだ。

Qwen-Drive-1.0-4B は、Qwen3.5-4B の VLM 本体に planning expert と BEV perception head を付けた構成で配布されている。
シーンラベリングに使うのは VLM 本体だけなので、init container で VLM の重みだけを取り出し、vLLM が読める Qwen3_5ForConditionalGeneration の checkpoint に変換した。
変換は tensor 名の先頭の vlm. を外すだけで、重みのバイト列は変更していない。
tensor 数は 723 で、Qwen3.5-4B の公開 checkpoint から MTP 層を除いた構成と一致する。
変換後のサーバーで benchmark Prompt を tokenize し、Qwen3.5-4B の tokenizer で作った token 列と一致することも確認した。

| 項目 | 値 |
|---|---|
| モデル | Qwen/Qwen-Drive-1.0-4B revision 28484089 |
| 推論サーバー | vllm/vllm-openai v0.32.0（CUDA 13） |
| GPU | g5.2xlarge（NVIDIA A10G 24GB）1 台、On-Demand USD 1.212 毎時 |
| 主な起動引数 | max-model-len 32768、max-num-batched-tokens 16384、trust-request-mm-kwargs、reasoning-parser qwen3 |
| client 側 | concurrency 8、request 順序 shuffled-cold（記事の Nano と同じ） |

記事の Nano と同じ g6e.2xlarge を使う予定だったが、us-west-2 の全 AZ で InsufficientInstanceCapacity が続いたため、起動できた g5.2xlarge を使った。
GPU の世代が異なるため、モデルの出力には影響しないが、応答時間と費用は記事の値と直接は比較できない。


4. 比較した条件

| 条件名 | 入力 | 判定方法 | 目的 |
|---|---|---|---|
| Qwen 12 frames | 記事と同一の Uniform 24 frames の mp4 | 13 タグ一括 | vLLM の既定動作のまま投入した場合 |
| Qwen Uniform 24 | 同じ 24 frames を 2 fps で再 encode | 13 タグ一括 | 主条件。Cosmos 3 Nano Uniform と比較 |
| Qwen Adaptive 24 | Adaptive 24 frames を 2 fps で再 encode | 13 タグ一括 | Cosmos 3 Super と同じ frame 選択で比較 |
| Qwen 単一タグ | Qwen Uniform 24 と同じ | 弱い 3 タグだけ 1 タグずつ判定 | 13 タグ同時判定の競合を外したときの改善 |
| Qwen thinking | Qwen Uniform 24 と同じ | 13 タグ一括、思考モード | 推論過程の生成で時系列タグが改善するか |
| VO prior | 上記の出力 | 記事の固定規則を適用 | 自車運動タグの補正 |

2 fps で再 encode した理由を述べる。
vLLM 0.32 の Qwen3-VL 系の video processor は、全 frame を読み込んだ動画を 2 fps に再 sampling する。
記事の mp4 は 24 frames を 4 fps で並べた 6 秒の動画なので、そのまま渡すと 12 frames しかモデルに届かない。
prompt token 数がこれを示しており、4 fps のままでは 7,263 tokens、同じ 24 frames を 2 fps で並べ直すと 11,777 tokens になる。
per-request の mm_processor_kwargs と video preprocessor config の fps は、この版の vLLM では効かなかった。
そこで、選択する frame と timestamp manifest を変えずに、表示 fps だけを 2 にした mp4 を作り直した。
Qwen 12 frames は、この補正をしない場合の比較条件である。

単一タグ条件で判定し直す 3 タグは、記事の 6.3 節で Cosmos 3 Super の F1 が 50% を下回ったタグをそのまま選んだ。
Prompt は通常条件と同じ Reasoned Prompt で、criteria を 1 件にし、GROUP-SPECIFIC RULES に記事のタグ群別 instruction を入れた。
閾値や Prompt を Evaluation の結果で調整してはいない。
ただし、3 タグの選択には記事の Evaluation 結果を使っているため、この改善を未知データでの性能として主張するには新しい holdout が要る。


5. 結果

5.1 全体指標

| モデルと条件 | Precision | Recall | F1 | Overall Accuracy | MCC | Coverage | USD 毎 1,000 clips |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosmos 3 Nano Adaptive 24 | 71.3% | 66.7% | 68.9% | 85.6% | 0.596 | 100.0% | 0.66 |
| Cosmos 3 Nano Uniform 24 | 73.4% | 65.9% | 69.4% | 86.1% | 0.606 | 100.0% | 0.69 |
| Cosmos 3 Super Adaptive 24 | 81.9% | 68.3% | 74.5% | 88.8% | 0.678 | 100.0% | 13.48 |
| Qwen 12 frames | 91.2% | 62.1% | 73.8% | 89.5% | 0.695 | 100.0% | 0.80 |
| Qwen Uniform 24 | 90.0% | 63.9% | 74.7% | 89.6% | 0.700 | 100.0% | 1.49 |
| Qwen Adaptive 24 | 89.6% | 64.0% | 74.7% | 89.6% | 0.699 | 100.0% | 1.76 |
| Qwen Uniform 24 + 単一タグ | 79.7% | 69.7% | 74.3% | 88.2% | 0.672 | 99.3% | 4.65 |
| Qwen Uniform 24 thinking | 84.2% | 63.8% | 72.6% | 81.5% | 0.658 | 90.5% | 約 2.6 |

Cosmos 3 の費用は記事の g6e.2xlarge と g6.24xlarge、Qwen-Drive の費用は g5.2xlarge の On-Demand 単価で計算した定常 GPU 費用である。

F1 差の 95% 信頼区間（source video block bootstrap）を示す。

| 比較 | F1 差 | 95% 信頼区間 |
|---|---:|---:|
| Qwen Uniform 24 と Cosmos 3 Nano Uniform | +5.3 | +3.9 から +6.8 |
| Qwen Uniform 24 と Cosmos 3 Super Adaptive | +0.2 | -1.1 から +1.6 |
| Qwen Adaptive 24 と Cosmos 3 Super Adaptive | +0.2 | -0.6 から +1.1 |
| Qwen Uniform 24 と Qwen 12 frames | +0.9 | +0.2 から +1.7 |
| Qwen Uniform 24 + 単一タグ と Qwen Uniform 24 | -0.4 | -1.6 から +0.7 |
| Qwen Uniform 24 thinking と Qwen Uniform 24 | -2.1 | -3.5 から -0.7 |

Qwen-Drive の改善は Precision 側に偏っている。
Cosmos 3 Super と比べて Precision は 8.1 points 高く、Recall は 4.4 points 低い。
記事の Super が主に False Positive を減らして Nano を上回ったのと同じ方向の変化が、4B のモデルでより強く出ている。

24 frames を全部入力すると、12 frames より F1 が 0.9 points 上がった。
差は小さいが信頼区間は 0 を上回っており、vLLM の既定の再 sampling を放置すると品質を少し失う。

5.2 性能の悪かったタグ

| タグ | GT positive | Cosmos 3 Nano Uniform | Cosmos 3 Super Adaptive | Qwen Uniform 24 | Qwen 単一タグ |
|---|---:|---:|---:|---:|---:|
| 車線変更 | 20 | 12.5 | 15.0 | 0.0 | 15.4 |
| 停止車両 | 272 | 53.5 | 44.9 | 37.7 | 53.9 |
| 歩行者横断待ち | 123 | 26.1 | 28.3 | 16.1 | 43.8 |
| 自車左折 | 61 | 27.8 | 59.8 | 32.7 | 判定し直していない |
| 歩行者横断 | 205 | 46.6 | 48.2 | 51.1 | 判定し直していない |

数値はタグ別 F1（%）である。
13 タグ一括の Qwen-Drive が弱いタグで落ちる理由は、positive をほとんど出さないことにある。
歩行者横断待ちは GT positive 123 件に対して positive 予測が 14 件しかなく、Precision 78.6%、Recall 8.9% だった。
車線変更は positive 予測が 2 件で、どちらも誤りだった。
停止車両も positive 予測は 94 件で、GT positive 272 件の 3 分の 1 程度にとどまる。

単一タグで判定し直すと、positive 予測が増えて Recall が上がった。

| タグ | 条件 | Precision | Recall | F1 | positive 予測数 |
|---|---|---:|---:|---:|---:|
| 車線変更 | 13 タグ一括 | 0.0 | 0.0 | 0.0 | 2 |
| 車線変更 | 単一タグ | 12.5 | 20.0 | 15.4 | 32 |
| 停止車両 | 13 タグ一括 | 73.4 | 25.4 | 37.7 | 94 |
| 停止車両 | 単一タグ | 48.0 | 61.4 | 53.9 | 348 |
| 歩行者横断待ち | 13 タグ一括 | 78.6 | 8.9 | 16.1 | 14 |
| 歩行者横断待ち | 単一タグ | 37.4 | 52.8 | 43.8 | 174 |

タグ別 F1 差の 95% 信頼区間を示す。

| タグ | 単一タグと 13 タグ一括 | 単一タグと Cosmos 3 Super | 単一タグと Cosmos 3 Nano |
|---|---:|---:|---:|
| 車線変更 | +4.5 から +25.4 | -12.6 から +17.6 | -10.0 から +17.7 |
| 停止車両 | +7.0 から +24.3 | +1.3 から +15.1 | -7.9 から +7.4 |
| 歩行者横断待ち | +10.5 から +39.8 | +0.1 から +27.2 | +9.9 から +23.5 |

歩行者横断待ちは、Cosmos 3 の Nano と Super の両方を上回った。
停止車両は Super を上回ったが、Nano とは差が確認できない。
車線変更は 13 タグ一括より改善したものの、Cosmos 3 と同じ 15% 前後にとどまり、GT positive が 20 件しかないため信頼区間も広い。

この改善は Recall と Precision の交換を伴う。
停止車両の Precision は 73.4% から 48.0% へ、歩行者横断待ちは 78.6% から 37.4% へ下がった。
記事の RQ6 で、タグ別 YES 判定が見逃しを回復する一方で False Positive も増やしたのと同じ傾向である。
13 タグを同時に判定させると、モデルは個々のタグの判定基準を厳しめに運用しており、1 タグずつ問うとその抑制が外れる、と解釈できる。
ただし、この解釈は positive 予測数の変化から推定したもので、モデル内部の機構を確かめたわけではない。

単一タグの request では 80 件（3.0%）の出力が parse できなかった。
多くは criteria が 1 件しかないのに FINAL_POSITIVE_INDICES: [10] のような存在しない index を返したもので、abstention として扱った。
13 タグ一括の答えで補った場合の F1 は、停止車両 56.1%、歩行者横断待ち 44.1% で、結論は変わらない。

5.3 弱いタグ以外の変化

自車左折は Qwen-Drive で 32.7% と、Cosmos 3 Super の 59.8% を大きく下回った（差の 95% 信頼区間は -37.7 から -15.8 points）。
自車右折も 37.1% で、Super の 55.8% を下回る。
一方、自転車存在 86.1%、対象信号赤 82.6%、対象信号青 74.9% は Super と同等だった。
物体と信号の状態は Qwen-Drive でも安定し、右左折のような自車の heading 変化は苦手、という分布になった。

5.4 固定 VO prior との組み合わせ

| 条件 | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|
| Cosmos 3 Super + VO prior | 82.8% | 70.5% | 76.2% | 0.698 |
| Qwen Uniform 24 + VO prior | 90.6% | 67.4% | 77.3% | 0.728 |
| Qwen Uniform 24 + VO prior + 単一タグ | 80.6% | 73.2% | 76.7% | 0.700 |

VO prior は記事の Development で固定した規則をそのまま使い、自車走行、自車停止、自車左折の 3 タグだけを変更する。
Qwen Uniform 24 に適用すると、改善 115 pairs、悪化 15 pairs で、F1 は +2.6 points（95% 信頼区間 +2.1 から +3.2）上がった。
Cosmos 3 Super + VO prior との差は +1.1 points で、95% 信頼区間 -0.2 から +2.5 は 0 を跨ぐ。
VO prior と単一タグは変更するタグが重ならないので組み合わせられるが、単一タグ側の Precision 低下で全体 F1 は 76.7% にとどまった。

5.5 思考モード

思考モードでは、Qwen-Drive は 1 clip 当たり平均 1,418 tokens を生成し、推論過程の文字数の中央値は約 3,200 文字だった。
84 clips（9.6%）は推論過程が max_tokens 4,096 に達し、続く repair request（64 tokens）でも思考モードのまま答えを出せず、13 タグすべてが abstention になった。
coverage 90.5% は、記事が方式の採用条件にした 99% を下回る。

| タグ | 思考モードなし F1 | 思考モード F1 | 差の 95% 信頼区間 |
|---|---:|---:|---:|
| 車線変更 | 0.0 | 0.0 | 0.0 から 0.0 |
| 停止車両 | 37.7 | 49.1 | +5.9 から +16.9 |
| 歩行者横断待ち | 16.1 | 36.2 | +4.4 から +32.2 |
| 自車左折 | 32.7 | 44.4 | +3.9 から +18.9 |

思考モードでも、停止車両、歩行者横断待ち、自車左折の F1 は上がった。
上がり幅は単一タグ条件より小さく、車線変更は positive 予測が 0 件のままだった。
答えを出せた 791 clips だけで比べると、思考モードの F1 は 77.4%、同じ clips での思考モードなしは 75.4% である。
ただし、上限に達した clip はおそらく判断の難しい clip に偏っており、この 2 points を思考モードの効果とみなすことはできない。
max_tokens を増やせば coverage は上がる見込みだが、生成 token がさらに増え、1 clip 当たりの時間は思考モードなしの約 1.8 倍からさらに伸びる。


6. 推論コストと運用上の注意

| 条件 | GPU | 1 clip 当たり wall time | USD 毎 1,000 clips | 走行動画 1 時間当たり |
|---|---|---:|---:|---:|
| Cosmos 3 Nano Uniform 24 | g6e.2xlarge | 1.1 s | 0.69 | 0.31 |
| Cosmos 3 Super Adaptive 24 | g6.24xlarge | 7.3 s | 13.48 | 6.07 |
| Qwen Uniform 24 | g5.2xlarge | 4.4 s | 1.49 | 0.67 |
| Qwen Uniform 24 + 単一タグ | g5.2xlarge | 13.8 s | 4.65 | 2.09 |
| Qwen Uniform 24 thinking | g5.2xlarge | 約 7.8 s | 約 2.6 | 約 1.18 |

走行動画 1 時間は 8 秒 clip 450 本として換算した。
Qwen-Drive は 4B と小さいが、1 clip 当たりの時間は Cosmos 3 Nano の 4 倍ほどかかった。
理由は三つある。
一つ目に、A10G は記事の L40S より演算性能が低い。
二つ目に、1 request の prompt が平均約 12,700 tokens あり、Cosmos 3 Nano の約 8,100 tokens より多い。
三つ目に、Qwen-Drive は 64 tokens の上限内で説明文を書き始めることがあり、Uniform 24 では 875 clips 中 378 clips（43%）、Adaptive 24 では 583 clips（67%）で repair request が発生した。
Cosmos 3 の 2 run では repair は一度も発生していない。
repair が発生した clip の F1 は 75.6%、発生しなかった clip は 73.8% で、repair 経由の答えが品質を落としてはいない。

Qwen3.5 は線形 attention と通常の attention を混ぜた構成で、vLLM 0.32 では Prompt prefix cache が 0% だった。
同じ動画に 3 タグを別 request で問う単一タグ条件では、動画の prefill を毎回やり直すことになり、費用が 3 倍強に増えた。

実験中に起きた serving の問題と対処を記録する。
Adaptive 24 の初回 run では、frame 数が奇数（21 枚、23 枚）の clip で動画の embedding が 12,432 から 13,200 tokens になり、vLLM が事前確保した encoder cache の 12,288 tokens を超えて stream の途中でエラーを返した。
benchmark client がこのエラーを空の応答として記録したため、coverage が 72.7% まで落ちた。
client が stream 内のエラーを失敗として記録するよう修正し、サーバーを max-num-batched-tokens 16384 で起動し直して、Adaptive 24 を最初から再実行した。
表の Qwen Adaptive 24 は再実行の結果で、coverage は 100% である。
Uniform の 2 run は動画 token が 12,288 に収まっており、初回からこの問題は起きていない（prompt token 数 0 の応答は 0 件）。
単一タグ条件はサーバー再起動の前に 60 request、後に 2,565 request を処理し、runner の resume で重複なく合計 2,625 request になった。


7. 妥当性の脅威と限界

GPU が記事と異なる。
品質指標はモデルとサーバー設定で決まり GPU には依存しないはずだが、応答時間と費用は A10G での値である。
L40S の g6e.2xlarge で測れば 1 clip 当たりの時間は短くなる見込みだが、未測定である。

入力 mp4 の表示 fps を 4 から 2 に変えた。
モデルに届く frame と timestamp manifest は記事と同じだが、Qwen3-VL が動画 token に付ける内部の timestamp は 0 秒から 11.5 秒の表示時刻になり、manifest の元動画時刻とは異なる。
Prompt は encode 後の再生速度が元の時刻と異なることを明示しているので、Cosmos 3 と同じ指示の下での比較になっている。

弱い 3 タグは記事の Evaluation 結果で選んだ。
単一タグ条件の Prompt と判定規則は事前に固定し、Evaluation を見て調整はしていないが、タグの選択に Evaluation の誤り分析を使っている。
単一タグの効果を一般化した主張にするには、Development で手順を固定したうえで新しい holdout で再評価する必要がある。

車線変更は GT positive が 20 件しかない。
タグ別 F1 の信頼区間が広く、Qwen-Drive と Cosmos 3 の差は判定できない。

Qwen-Drive の planning expert と BEV perception head は使っていない。
Qwen-Drive の自車軌跡予測は自車運動タグ、特に右左折と車線変更の補助証拠になりうるが、ROAD の前方単眼映像は Qwen-Drive の planning 入力（複数 camera と ego status）を満たさないため、今回の比較から外した。

Development 169 clips では Qwen-Drive を評価していない。
記事と同じく Development で方式を選び Evaluation で一度だけ評価する手順を Qwen-Drive について踏んだわけではなく、今回の数値は Cosmos 3 用に固定した Prompt をそのまま移したときの性能である。


8. 次に試すこと

単一タグの効果は Recall と Precision の交換なので、YES と NO の log probability から score を取り、Development で閾値を固定すれば、Precision の低下を抑えられる可能性がある。
記事の RQ6 の YES score 診断を、Qwen-Drive の速さを使って Development 169 clips 全体で行うのが最初の候補になる。
右左折の低下は、VO prior の turn veto を右折にも広げる規則を Development で探索すると改善するかを確かめたい。


付録 A 成果物の場所

| 内容 | 場所 |
|---|---|
| 復元した benchmark コードと Qwen-Drive 用の変更 | qwen-drive-extension/benchmark |
| run の実行スクリプト | qwen-drive-extension/scripts |
| 評価と分析のスクリプト | qwen-drive-extension/tools |
| 集計済みの評価結果 | qwen-drive-extension/results |
| EKS manifest と checkpoint 変換スクリプト | riita10069/cosmos3-nano-vllm-eks-poc の qwen-drive |

ROAD の動画、encode 済みの入力動画、clip ごとの予測は、ROAD と Oxford RobotCar のライセンス上の制約があるため含めていない。
