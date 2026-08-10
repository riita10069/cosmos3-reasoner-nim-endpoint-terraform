# 敵対的自己査読記録

本記録は、論文生成時に実施した三巡の敵対的自己査読と、その解決状況を示す。
独立査読者による査読を代替するものではない。

## 第1巡: 科学的妥当性

- **指摘:** Hybrid CoreのDevelopment結果、GTを使う48-frame Oracle診断、
  Evaluation Full Runが同一水準に見える。
  **解決:** 証拠水準表を追加し、Population evaluation、Development selection、
  Oracle diagnostic、Production proposalを分離した。
- **指摘:** 既存稿の「Oracleでも改善しない」と、Super Oracle 48の5/14回復が矛盾する。
  **解決:** GTを使わないmotion samplingは不調だが、難例限定Oracle 48は一部を回復した、
  という限定的結論へ修正した。
- **指摘:** YES/NO log probabilityを校正済みconfidenceと誤読できる。
  **解決:** 全文を「未校正YES score」へ変更し、独立calibrationの必要性を明記した。
- **指摘:** 閾値0.05の11/14回復だけを強調するとcontrol FP 5/7を隠す。
  **解決:** 7--11/14回復とcontrol FP 2--5/7のtrade-offを併記した。

## 第2巡: 方法・統計・再現性

- **指摘:** Developmentは3 source videosで、candidate比較の多重性もある。
  **解決:** block bootstrapのcluster数が少ないこと、多重比較補正をしていないこと、
  固定Evaluationが必要なことを限界へ追記した。
- **指摘:** GT event時刻・GT boxを使った診断結果を本番方式の性能へ流用できない。
  **解決:** Oracle結果は原因分解と上限診断に限定し、実運用候補生成器の性能を未測定とした。
- **指摘:** 48-frame費用は全clip費用ではない。
  **解決:** 表を1,000 routed tag checks当たりのqueue-adjusted proxyと明記した。
- **指摘:** 入力・結果の追跡可能性が必要。
  **解決:** すべての入力を提出directoryへ同梱し、SHA-256 manifestと再生成手順を出力した。

## 第3巡: 明瞭性・組版・実務上の解釈

- **指摘:** 推奨構成が「実証済み」と「次に検証する提案」を混同する。
  **解決:** Evaluation実証済み構成、Development選定extension、Oracle由来extensionを分けた。
- **指摘:** 「新規FP」はbaseline controlにも使われ、意味が不正確である。
  **解決:** 表記をnegative-control FPへ統一した。
- **指摘:** タグごとの次の実装が抽象的である。
  **解決:** 低Recall 7タグについて、診断上の制約と本番で近似すべき構造化証拠を表にした。
- **指摘:** 表・段組み・日本語glyph・参照の破綻可能性がある。
  **解決:** TeX warning、欠落glyph、未定義参照、全ページ画像を最終監査対象とした。
