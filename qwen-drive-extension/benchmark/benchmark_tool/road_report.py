from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .util import load_json, sha256_file, utc_now, write_json


TAG_LABELS = {
    "ego_moving": "自車走行",
    "ego_stopped": "自車停止",
    "ego_turn_left": "自車左折",
    "ego_turn_right": "自車右折",
    "ego_lane_change": "自車車線変更",
    "car_braking": "車両制動",
    "car_stopped": "停止車両",
    "pedestrian_waiting_to_cross": "歩行者横断待ち",
    "pedestrian_crossing": "歩行者横断",
    "cyclist_present": "自転車存在",
    "motorcycle_present": "二輪車存在",
    "traffic_light_red": "対象信号赤",
    "traffic_light_green": "対象信号青",
}


def generate_road_paper(
    evaluation_summary_path: Path,
    development_summary_path: Path,
    smoke_summary_path: Path,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    compile_pdf: bool,
    error_analysis_path: Path | None = None,
    sampling_audit_path: Path | None = None,
    serving_repeat_path: Path | None = None,
    cascade_path: Path | None = None,
) -> tuple[Path, ...]:
    evaluation = load_json(evaluation_summary_path)
    development = load_json(development_summary_path)
    smoke = load_json(smoke_summary_path)
    manifest = load_json(manifest_path)
    audit = load_json(audit_path)
    error_analysis = (
        load_json(error_analysis_path) if error_analysis_path else None
    )
    sampling_audit = (
        load_json(sampling_audit_path) if sampling_audit_path else None
    )
    serving_repeat = (
        load_json(serving_repeat_path) if serving_repeat_path else None
    )
    cascade = load_json(cascade_path) if cascade_path else None
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_inputs = {
        "evaluation-summary.json": evaluation_summary_path,
        "development-summary.json": development_summary_path,
        "serving-smoke-summary.json": smoke_summary_path,
        "road-benchmark-manifest.json": manifest_path,
        "road-annotation-audit.json": audit_path,
    }
    optional_inputs = {
        "error-analysis.json": error_analysis_path,
        "sampling-audit.json": sampling_audit_path,
        "serving-repeat-summary.json": serving_repeat_path,
        "cascade-summary.json": cascade_path,
    }
    source_inputs.update(
        {
            name: path
            for name, path in optional_inputs.items()
            if path is not None
        }
    )
    bundled_inputs: dict[str, Path] = {}
    for name, source in source_inputs.items():
        destination = input_dir / name
        shutil.copy2(source, destination)
        bundled_inputs[name] = destination

    tex_path = output_dir / "ROAD_COSMOS3_SCENE_LABELING_PAPER.tex"
    markdown_path = output_dir / "ROAD_COSMOS3_SCENE_LABELING_PAPER.md"
    reproduction_path = output_dir / "REPRODUCTION.md"
    tex_path.write_text(
        _render_tex(
            evaluation,
            development,
            smoke,
            manifest,
            audit,
            error_analysis,
            sampling_audit,
            serving_repeat,
            cascade,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_markdown(
            evaluation,
            development,
            smoke,
            manifest,
            audit,
            error_analysis,
            sampling_audit,
            serving_repeat,
            cascade,
        ),
        encoding="utf-8",
    )
    reproduction_path.write_text(
        _render_reproduction(
            bundled_inputs["evaluation-summary.json"],
            bundled_inputs["development-summary.json"],
            bundled_inputs["serving-smoke-summary.json"],
            bundled_inputs["road-benchmark-manifest.json"],
            bundled_inputs["road-annotation-audit.json"],
            output_dir,
            supplemental_paths={
                name: bundled_inputs[name]
                for name in optional_inputs
                if name in bundled_inputs
            },
        ),
        encoding="utf-8",
    )

    pdf_path: Path | None = None
    if compile_pdf:
        tectonic = shutil.which("tectonic")
        if not tectonic:
            raise FileNotFoundError("tectonic is required to compile the PDF")
        subprocess.run(
            [
                tectonic,
                "--keep-logs",
                "--outdir",
                str(output_dir),
                str(tex_path),
            ],
            check=True,
        )
        pdf_path = output_dir / "ROAD_COSMOS3_SCENE_LABELING_PAPER.pdf"
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)

    artifacts = [
        tex_path,
        markdown_path,
        reproduction_path,
        *([pdf_path] if pdf_path else []),
    ]
    manifest_output = output_dir / "road-paper-artifact-manifest.json"
    write_json(
        manifest_output,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "inputs": [
                {
                    "path": str(bundled_inputs[name].relative_to(output_dir)),
                    "source_path": str(source.resolve()),
                    "sha256": sha256_file(bundled_inputs[name]),
                }
                for name, source in source_inputs.items()
            ],
            "artifacts": [
                {"path": path.name, "sha256": sha256_file(path)}
                for path in artifacts
                if path is not None
            ],
        },
    )
    return tuple([*artifacts, manifest_output])


def _render_tex(
    evaluation: dict[str, Any],
    development: dict[str, Any],
    smoke: dict[str, Any],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    error_analysis: dict[str, Any] | None = None,
    sampling_audit: dict[str, Any] | None = None,
    serving_repeat: dict[str, Any] | None = None,
    cascade: dict[str, Any] | None = None,
) -> str:
    evaluation_profiles = evaluation["profiles"]
    development_profiles = development["profiles"]
    smoke_profiles = smoke["profiles"]
    best_eval = max(
        evaluation_profiles,
        key=lambda row: float(row["micro"]["strict_f1"] or -1),
    )
    best_development = max(
        development_profiles,
        key=lambda row: float(row["micro"]["strict_f1"] or -1),
    )
    eval_rows = "\n".join(
        _quality_row(row) for row in evaluation_profiles
    )
    dev_rows = "\n".join(
        _ablation_row(row)
        for row in sorted(
            development_profiles,
            key=lambda item: float(item["micro"]["strict_f1"] or -1),
            reverse=True,
        )
    )
    algorithm_runtime_rows = "\n".join(
        _runtime_row(row)
        for row in _algorithm_smoke_profiles(smoke_profiles)
    )
    serving_runtime_rows = "\n".join(
        _runtime_row(row)
        for row in _serving_sweep_profiles(smoke_profiles)
    )
    tag_rows = _tag_comparison_rows(evaluation_profiles)
    distribution_rows = "\n".join(
        _distribution_row(row) for row in evaluation_profiles
    )
    curation_rows = _curation_rows(evaluation_profiles)
    ci = best_eval.get("bootstrap_ci95") or {}
    f1_ci = ci.get("strict_f1")
    eval_clip_count = max(
        int(row.get("clip_count", 0)) for row in evaluation_profiles
    )
    eval_video_count = max(
        int(row.get("source_video_count", 0))
        for row in evaluation_profiles
    )
    split_counts = manifest["split_clip_counts"]
    known_pairs = int(best_eval["known_pairs"])
    unknown_pairs = (
        int(split_counts["evaluation"]) * len(manifest["tags"])
        - known_pairs
    )
    best_name = _profile_label(best_eval)
    best_f1 = _pct(best_eval["micro"]["strict_f1"])
    best_mcc = _number(best_eval["micro"]["selective_mcc"])
    best_cost = _number(
        best_eval["runtime"]["estimated_usd_per_1000_clips"], 2
    )
    f1_ci_text = (
        f"{_pct(f1_ci[0])}--{_pct(f1_ci[1])}" if f1_ci else "算出不能"
    )
    tags_text = "、".join(
        _tex(TAG_LABELS.get(tag["id"], tag["id"]))
        for tag in manifest["tags"]
    )
    model_comparison = _model_comparison_text(evaluation_profiles)
    tag_findings = _tag_findings_text(best_eval)
    cost_projection = _cost_projection_text(evaluation_profiles)
    serving_findings = _road_serving_findings_text(smoke_profiles)
    supplemental_results = _supplemental_results_tex(
        error_analysis,
        sampling_audit,
        serving_repeat,
        cascade,
    )
    template = r"""\documentclass[9pt,a4paper]{extarticle}
\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,siunitx,graphicx}
\usepackage[margin=16mm,columnsep=7mm]{geometry}
\usepackage{hyperref,titlesec,caption,multicol,ragged2e,needspace}
\defaultfontfeatures{Ligatures=TeX}
\IfFontExistsTF{TeX Gyre Termes}{
  \setmainfont{TeX Gyre Termes}
  \setsansfont{TeX Gyre Heros}
}{
  \setmainfont{Times New Roman}
  \setsansfont{Helvetica Neue}
}
\setmonofont{Menlo}
\IfFontExistsTF{Hiragino Mincho ProN}{
  \setCJKmainfont{Hiragino Mincho ProN}
  \setCJKsansfont{Hiragino Sans}
  \setCJKmonofont{Hiragino Sans}
}{
  \IfFontExistsTF{Noto Serif CJK JP}{
    \setCJKmainfont{Noto Serif CJK JP}
    \setCJKsansfont{Noto Sans CJK JP}
    \setCJKmonofont{Noto Sans CJK JP}
  }{
    \setCJKmainfont{Amazon Ember JP}
    \setCJKsansfont{Amazon Ember JP}
    \setCJKmonofont{Amazon Ember JP}
  }
}
\hypersetup{
  unicode=true,
  colorlinks=false,
  pdfborder={0 0 0},
  pdftitle={公開自動運転動画に対する視覚言語モデルを用いた時系列シーンラベリング}
}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\columnsep}{7mm}
\setlength{\tabcolsep}{2.6pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.08}
\titleformat{\section}{\normalsize\bfseries\sffamily}{\thesection.}{0.45em}{}
\titleformat{\subsection}{\small\bfseries\sffamily}{\thesubsection}{0.4em}{}
\captionsetup{font=small,labelfont=bf}
\newcommand{\modelname}[1]{\texttt{\small #1}}
\renewcommand{\abstractname}{概要}
\renewcommand{\tablename}{表}
\renewcommand{\figurename}{図}
\renewcommand{\refname}{参考文献}

\title{\bfseries 公開自動運転動画に対する視覚言語モデルを用いた\\
時系列シーンラベリングの品質・計算費用評価\\[2pt]
\large Quality and Computational Cost of Temporal Scene Labeling\\
with a Vision-Language Model on Public Driving Videos}
\author{Ryota Yamada\\{\small Amazon Web Services, Inc.}}
\date{}

\begin{document}
\maketitle
\begin{abstract}
自動運転用走行データのラベリングは、学習データの分布把握、希少事象の検索、
hard-negative mining、およびactive learningにおける人手確認対象の選定に必要である。
一方、既存の大規模走行データでは負例が網羅的に確認されていない場合があり、
PrecisionやAccuracyを含む定量評価が成立しない。本稿では、密な時間注釈を持つ公開
ROAD datasetを用い、8秒clipに対する13タグの完全な正負Ground Truthを構成した。
元動画単位でdevelopment/evaluationを分離し、Cosmos 3 NanoおよびSuperについて、
frame sampling、timestamp明示、prompt形式、grouped request、multimodal/prefix cacheが
品質、tail latency、stage時間およびGPU費用へ与える影響を評価した。
同一ROAD clipsを用いるcontrolled serving experimentにより、continuous batching、
Chunked Prefillおよびcache再利用の効果を、入力内容とpromptを固定して分離した。
evaluationは@@EVAL_CLIPS@@ clips、@@EVAL_VIDEOS@@ source videos、
@@KNOWN_PAIRS@@ known clip--tag pairsを含む。最良条件@@BEST_NAME@@は
micro F1 @@BEST_F1@@（source-video block bootstrap 95\% CI:
@@F1_CI@@）、MCC @@BEST_MCC@@、推定費用
\$@@BEST_COST@@/1,000 clipsであった。@@MODEL_COMPARISON@@
さらに、予測ラベルによる分布推定誤差と
希少シーンの人手確認効率を評価し、自動ラベリングの利用可能範囲と限界を示す。
\end{abstract}

\noindent\textbf{キーワード:} 自動運転、視覚言語モデル、動画理解、シーンラベリング、
active learning、データキュレーション

\begin{multicols}{2}
\section{はじめに}
自動運転システムの開発では、長時間の走行動画から学習・検証に有用な区間を抽出し、
データ集合の偏りやlong-tail scenarioを把握する必要がある。物体検出boxのような
局所注釈だけでなく、「自車が車線変更した」「歩行者が横断を待っている」
「先行車が制動した」といった時間的scene labelは、データ検索、失敗分析、
評価set設計およびactive learningのacquisitionに利用できる。しかし、全動画を
人手で網羅的に確認する費用は大きく、正例探索だけを目的とした弱いGround Truthでは
false positiveを測れないため、モデルがyesを過剰出力しても高く評価され得る。

本研究の目的は、(i) 完全な正負GT上でVision-Language Model（VLM）の時系列
multi-label分類能力を測ること、(ii) 入力frameとrequest構成が品質・latency・費用へ
及ぼす影響を分離すること、(iii) 生成ラベルが分布推定と希少scene抽出に与える
実用的価値を測ることである。主な貢献は次の3点である。
\begin{enumerate}
  \item ROADの密な注釈から、unknownを明示した再現可能な13タグ・8秒clip benchmarkを構築した。
  \item timestamp付きsampling、evidence重視prompt、grouped requestを同一media bytesで比較し、
        source-video単位のblock bootstrapにより不確実性を評価した。
  \item 分類品質だけでなく、分布誤差、希少scene retrieval、tail latency、
        prefill/decode、cache hit、GPU費用を、ROAD上の再現可能なartifactから報告した。
\end{enumerate}

\section{関連研究}
\subsection{道路事象dataset}
ROADはOxford RobotCar動画に対し、agent、action、location、ego actionおよび
複合road eventをframe/track単位で注釈したdatasetであり、online event detectionや
anticipationを目的としている\cite{singh2022road,maddern2017robotcar}。
本稿はbounding-box detectionを再学習するのではなく、公開注釈をclip-levelの
multi-label GTへ決定論的に変換し、汎用VLMによるoff-line labelingを評価する。

\subsection{VLMと動画推論}
Cosmos 3はlanguage、image、video、audio、actionを統一architectureで扱う
omnimodal world model familyである\cite{nvidia2026cosmos3}。本研究は同familyの
Nano/Superを対象とし、モデル規模だけでなく入力時系列とserving構成を比較する。
大量video tokenの処理ではprefillが支配的になり得るため、PagedAttention、
continuous batchingおよびprefix reuseを実装するvLLM\cite{kwon2023vllm}上で計測した。

\subsection{Active learningとデータキュレーション}
自動運転におけるactive learningは、限られた注釈予算で情報量・多様性の高いframeを
選択する問題として研究されている\cite{lin2024active,bengar2019temporal}。
本稿ではdownstream modelの再学習効果を直接主張せず、VLM予測で人手確認対象を
優先したときのPrecision@K、Recall@K、random sampling比enrichmentをproxyとして測る。

\section{問題設定と評価指標}
clip集合を$\mathcal{V}$、タグ集合を$\mathcal{T}$とする。既知GTを
$y_{v,t}\in\{0,1\}$、モデル出力を
$\hat y_{v,t}\in\{0,1,\bot\}$とし、$\bot$はparse failureまたは欠落出力である。
unknown GT pairは母数から除外するが、モデルの$\bot$は正例に対してfalse negative相当、
負例に対して不正解としてstrict指標へ含める。主指標は
\[
  P=\frac{TP}{TP+FP},\quad
  R=\frac{TP}{TP+FN+A_+},\quad
  F_1=\frac{2PR}{P+R},
\]
ここで$A_+$は陽性GTに対するabstention数である。class imbalanceに対し
Balanced Accuracy（BA）とMatthews Correlation Coefficient（MCC）も報告する。
出力pair coverageが99\%未満の条件は品質比較から無効とする。

dataset分布推定はタグ別prevalenceのMean Absolute Error（MAE）と
Jensen--Shannon divergence（JSD）で測る。統計的区間は時間的に近接したclipを
独立と仮定せず、source videoを再標本化単位とする10,000回のblock bootstrapで得る。

\section{公開GTの構築}
ROAD train/validation 18動画の@@ANNOTATED_FRAMES@@ annotated frames
（全@@ANNOTATION_FRAMES@@ framesの@@ANNOTATED_RATIO@@）を使用した。
8秒window、8秒stride、12 FPS原時刻で1,044 clipsを生成し、ROAD公式fold 3に従って
development @@DEV_CLIPS@@ clips（3動画）とevaluation @@TEST_CLIPS@@ clips
（15動画）へ分離した。各タグは6 annotated frames以上、存在系は3 frames以上を
positiveとし、全frameが注釈済みで対象eventがないclipをnegative、境界欠損を
unknownとした。evaluationのunknownは@@UNKNOWN_PAIRS@@ pairsである。
使用タグは@@TAGS@@である。

\section{手法}
\subsection{Evidence media}
各8秒clipから共通evidence mediaを作る。uniform条件は1--8 FPS、adaptive条件は
全域の低rate sampleにframe差分・optical flowのpeak近傍を加え、最大24 framesとした。
選択frameには元動画timestampをmanifestとして付与した。timestamp有無やprompt違いの
条件ではmediaを再encodeせず、同一SHA-256 bytesをhard linkで共有した。

\begin{center}
\small
\fbox{ROAD video}
$\rightarrow$
\fbox{timestamp付きevidence media}
$\rightarrow$
\fbox{Cosmos 3}
$\rightarrow$
\fbox{13タグ}
$\rightarrow$
\fbox{分布推定・人手確認}
\end{center}

\subsection{Promptとrequest構成}
標準promptは全13基準をordered binary arrayとして返す。reasoned条件は全frameと
timestampを確認した後、positive indexだけを短く返す。grouped条件は同一mediaと
system prefixを保ち、ego action、road-user action、object/signalの3 requestに分ける。
同一clipのrequestを連続実行し、multimodal cacheとprefix cacheを再利用する。
ordered binary arrayは初期Smokeの64 tokensから正式runで128 tokensへ拡張した。
reasoned条件はpositive indexだけを返すため、全runで64 tokensとした。

\section{実験設定}
developmentでは15 clipsのclass-balanced Smokeで明らかに劣る条件を除外し、その後
全169 clipsでsampling/promptを選定した。evaluation 875 clipsは条件固定後に一度だけ
Nano/Superへ適用した。temperature 0、seed 20260808とし、モデル、prompt、media、
container digest、instance、GPU telemetryおよび全SHA-256をlock artifactへ保存した。
Nanoはg6e.2xlarge（1 GPU）、Superはg6.24xlarge（4 GPU）で実行した。後者は
GPU間がPCIe-onlyでvLLM custom all-reduceを利用できず、NCCL fallbackを含む実測である。
EC2定常推論費用は、artifactに記録したon-demand単価$p$とitem当たりwall時間$t$から
\[
 C_{1000}=p\frac{t}{3600}\times 1000
\]
で算出した。観測単価はus-west-2におけるg6e.2xlargeの
\$2.24208/hとg6.24xlargeの\$6.6752/hである。EKS control plane、EBS、NAT、
data transfer、model download、compile、idle時間は含めない。

\section{実験結果}
\subsection{Development ablation}
表\ref{tab:ablation}は全development splitの条件選定結果であり、evaluation性能ではない。
\begin{center}
\centering
\captionof{table}{Development splitにおける入力・prompt ablation}
\label{tab:ablation}
\scriptsize
\begin{tabular}{lrrrrr}
\toprule
条件 & P & R & F1 & BA & MCC\\
\midrule
@@DEV_ROWS@@
\bottomrule
\end{tabular}
\end{center}

\subsection{Evaluation Full Run}
表\ref{tab:quality}は固定した同一@@EVAL_CLIPS@@ clipsに対するFull Runのみを示す。
\end{multicols}
\begin{center}
\centering
\captionof{table}{Evaluation splitにおける完全GT品質（Full Runのみ）}
\label{tab:quality}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Model/条件 & clips & coverage & Precision & Recall & F1 & Accuracy & BA & MCC\\
\midrule
@@EVAL_ROWS@@
\bottomrule
\end{tabular}
}
\end{center}

\begin{center}
\centering
\captionof{table}{タグ別F1。``--''は陽性予測または有効分母がなくF1を定義できない場合を示す}
\label{tab:tags}
\small
\begin{tabular}{lrrr}
\toprule
タグ & GT陽性 & @@MODEL_HEADERS@@\\
\midrule
@@TAG_ROWS@@
\bottomrule
\end{tabular}
\end{center}
\begin{multicols}{2}

\subsection{Evidence・prompt構成と実行時間}
表\ref{tab:algorithm-runtime}はROAD development Smokeにおける入力・request構成の
計測であり、品質の最終比較には用いない。同一clipと同一model上でuniform sampling、
adaptive evidence、reasoned output、3-group requestを比較した。grouped条件では
同一mediaのfollow-upを連続実行し、multimodal/prefix cacheを再利用した。
この選定runはprofileをclipごとに連続する\texttt{video-grouped-hot}順序で実行したため、
adaptive行は先行profileのMM cacheを利用し得る。したがって表\ref{tab:algorithm-runtime}は
stageとcacheの観測に用い、profile間のcold latency比較には用いない。
\end{multicols}
\begin{center}
\centering
\captionof{table}{ROAD Smokeにおけるevidence・prompt構成別の実行時間}
\label{tab:algorithm-runtime}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Model/条件 & req/clip & wall/clip[s] & P95[s] & prefill[s] & decode[s] &
MM hit & Prefix hit & preempt. & USD/1k\\
\midrule
@@ALGORITHM_RUNTIME_ROWS@@
\bottomrule
\end{tabular}
}
\end{center}
\begin{multicols}{2}

\subsection{制御されたServing最適化実験}
表\ref{tab:serving-sweep}は、ROADの同一15 clips、adaptive 24-frame media、
reasoned prompt、temperature 0を固定したserving sweepである。Nanoは各pointで
podを再作成し、server defaultのconcurrency 1/2/4/8と、concurrency 4における
Chunked Prefill budget 2,048/8,192 tokensを比較した。Superはconcurrency 4/8を
比較した。これらは少数Smokeであるため、latencyの一般的な母平均ではなく、
本構成における性能境界として解釈する。costはon-demand時間単価とrun wall timeから
算出し、storage、control plane、model download、compile、idle startupを除く。
\end{multicols}
\begin{center}
\centering
\captionof{table}{ROAD同一15 clipsによるserving controlled sweep}
\label{tab:serving-sweep}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
Model/条件 & req/clip & wall/clip[s] & P95[s] & prefill[s] & decode[s] &
MM hit & Prefix hit & preempt. & USD/1k\\
\midrule
@@SERVING_RUNTIME_ROWS@@
\bottomrule
\end{tabular}
}
\end{center}
\begin{multicols}{2}

@@SERVING_FINDINGS@@

@@SUPPLEMENTAL_RESULTS@@

\subsection{分布推定と希少scene抽出}
\begin{center}
\centering
\captionof{table}{生成ラベルによるdataset分布推定}
\label{tab:distribution}
\scriptsize
\begin{tabular}{lrr}
\toprule
条件 & prevalence MAE & JSD\\
\midrule
@@DISTRIBUTION_ROWS@@
\bottomrule
\end{tabular}
\end{center}

@@CURATION_BLOCK@@

\section{考察}
\subsection{ラベリングの意義}
scene labelは単なる検索metadataではない。第一に、タグprevalenceにより収集datasetの
偏りを監視できる。第二に、希少event候補をrankし、人手確認をrandom samplingより
集中できる。第三に、model disagreementや低信頼区間をactive learning候補として、
downstream perception/planning modelの再学習setへ接続できる。ただし本稿の
retrieval評価は人手確認効率のproxyであり、再学習後の性能向上を直接示すものではない。

\subsection{品質と費用の解釈}
最良条件は@@BEST_NAME@@で、F1 @@BEST_F1@@、MCC @@BEST_MCC@@であった。
@@MODEL_TRADEOFF@@
frame数を増やすだけでは性能は単調増加せず、時間基準を明示したprompt形式の影響が
大きかった。grouped requestはcall数を増やす一方、同一mediaを再利用するため
cache hitが高く、細粒度基準のrecall改善とwall timeのtrade-offを持つ。
@@TAG_FINDINGS@@

\subsection{運用費用の感度}
@@COST_PROJECTION@@

\subsection{再現性と実験妥当性}
品質評価は公開ROADだけから構成し、source video単位でdevelopment/evaluationを分離した。
evaluation条件はdevelopmentで固定後に一度だけ適用し、serving Smokeの品質値を
Full Run表へ混入させていない。全runは入力media、timestamp、prompt、container digest、
vLLM command、pod profile、GPU telemetryおよびSHA-256を保存した。Table
\ref{tab:serving-sweep}は同一15 clipsに対する各1回の探索的計測であり、
latencyの信頼区間やinstance間変動を推定していない。この制約を越えて
小数点以下の差を一般化せず、採否は大きなthroughput差、tail latencyおよび
preemptionの有無に基づけた。

\subsection{限界}
ROADは英国Oxford周辺の限定されたcamera/domainであり、天候・地域・sensor placementの
一般化を保証しない。clip-level変換はevent onset/offsetのtemporal localization精度を
直接測らない。またbinary出力には連続confidence scoreがないためPR-AUCを算出していない。
希少タグはsource video数が少なくbootstrap CIが広い。Superの速度はPCIe-only 4 GPU
構成の制約を含み、NVLink/NVSwitch環境の上限性能ではない。これらを越える主張は行わない。

\section{結論}
本稿は、公開ROAD datasetの密な時間注釈から完全な正負GTを構成し、Cosmos 3による
時系列scene labelingを品質、計算費用、serving効率、分布推定および希少scene抽出の
観点から評価した。正例のみを信用する評価を避け、false positiveとabstentionを含む
confusion matrix、元動画単位CI、99\% coverage gateを導入した点が重要である。
結果は、入力frame数よりもtimestampを含む判定手順とprompt構成が品質へ大きく影響し、
同一mediaを用いる複数requestではcache再利用により追加費用を抑えられることを示した。
また、ROAD上のcontrolled serving実測から、continuous batching、Chunked Prefill、
短いoutput、同一mediaのMM/prefix cache再利用が、互いに異なるstageへ作用することを
確認した。したがって、モデル選択だけでなく、evidence生成からrequest schedulingまでを
一体で設計する必要がある。
今後は、異なる地域の公開dataset、event境界のtemporal mAP、連続confidence、
および実際のdownstream再学習を含むactive learning loopで検証する。

\begin{thebibliography}{9}
\bibitem{singh2022road}
G. Singh et al., ``ROAD: The ROad event Awareness Dataset for Autonomous Driving,''
\textit{IEEE Trans. Pattern Anal. Mach. Intell.}, 2022,
doi:10.1109/TPAMI.2022.3150906.
\bibitem{maddern2017robotcar}
W. Maddern et al., ``1 Year, 1000 km: The Oxford RobotCar Dataset,''
\textit{Int. J. Robotics Research}, 2017.
\bibitem{nvidia2026cosmos3}
NVIDIA et al., ``Cosmos 3: Omnimodal World Models for Physical AI,''
arXiv:2606.02800, 2026.
\bibitem{kwon2023vllm}
W. Kwon et al., ``Efficient Memory Management for Large Language Model Serving
with PagedAttention,'' in \textit{Proc. ACM SOSP}, 2023.
\bibitem{lin2024active}
J. Lin et al., ``Exploring Diversity-based Active Learning for 3D Object Detection
in Autonomous Driving,'' \textit{IEEE Trans. Intell. Transp. Syst.}, 2024.
\bibitem{bengar2019temporal}
J. Z. Bengar et al., ``Temporal Coherence for Active Learning in Videos,''
in \textit{Proc. ICCV Workshops}, 2019.
\end{thebibliography}
\end{multicols}
\end{document}
"""
    replacements = {
        "@@EVAL_CLIPS@@": str(eval_clip_count),
        "@@EVAL_VIDEOS@@": str(eval_video_count),
        "@@KNOWN_PAIRS@@": f"{known_pairs:,}",
        "@@BEST_NAME@@": _tex(best_name),
        "@@BEST_F1@@": best_f1,
        "@@F1_CI@@": f1_ci_text,
        "@@BEST_MCC@@": best_mcc,
        "@@BEST_COST@@": best_cost,
        "@@MODEL_COMPARISON@@": model_comparison["abstract"],
        "@@MODEL_TRADEOFF@@": model_comparison["discussion"],
        "@@TAG_FINDINGS@@": tag_findings,
        "@@COST_PROJECTION@@": cost_projection,
        "@@ANNOTATED_FRAMES@@": f"{int(audit['annotated_frame_count']):,}",
        "@@ANNOTATION_FRAMES@@": f"{int(audit['annotation_frame_count']):,}",
        "@@ANNOTATED_RATIO@@": _pct(
            int(audit["annotated_frame_count"])
            / int(audit["annotation_frame_count"])
        ),
        "@@DEV_CLIPS@@": str(split_counts["development"]),
        "@@TEST_CLIPS@@": str(split_counts["evaluation"]),
        "@@UNKNOWN_PAIRS@@": str(unknown_pairs),
        "@@TAGS@@": tags_text,
        "@@DEV_ROWS@@": dev_rows,
        "@@EVAL_ROWS@@": eval_rows,
        "@@MODEL_HEADERS@@": " & ".join(
            _tex(_profile_label(row)) for row in evaluation_profiles
        ),
        "@@TAG_ROWS@@": tag_rows,
        "@@ALGORITHM_RUNTIME_ROWS@@": algorithm_runtime_rows,
        "@@SERVING_RUNTIME_ROWS@@": serving_runtime_rows,
        "@@SERVING_FINDINGS@@": serving_findings,
        "@@SUPPLEMENTAL_RESULTS@@": supplemental_results,
        "@@DISTRIBUTION_ROWS@@": distribution_rows,
        "@@CURATION_BLOCK@@": curation_rows,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _quality_row(row: dict[str, Any]) -> str:
    micro = row["micro"]
    return (
        f"{_tex(_profile_label(row))} & {row['clip_count']} & "
        f"{_pct(row['output_pair_coverage'])} & "
        f"{_pct(micro['precision'])} & {_pct(micro['strict_recall'])} & "
        f"{_pct(micro['strict_f1'])} & "
        f"{_pct(micro['overall_accuracy'])} & "
        f"{_pct(micro['strict_balanced_accuracy'])} & "
        f"{_number(micro['selective_mcc'])} \\\\"
    )


def _ablation_row(row: dict[str, Any]) -> str:
    micro = row["micro"]
    validity = "" if row["output_validity"]["passed"] else "$^{\\dagger}$"
    return (
        f"{_tex(_short_profile(row['profile']))}{validity} & "
        f"{_pct(micro['precision'])} & {_pct(micro['strict_recall'])} & "
        f"{_pct(micro['strict_f1'])} & "
        f"{_pct(micro['strict_balanced_accuracy'])} & "
        f"{_number(micro['selective_mcc'])} \\\\"
    )


def _runtime_row(row: dict[str, Any]) -> str:
    runtime = row["runtime"]
    requests_per_clip = (
        float(runtime["request_count"])
        / float(runtime.get("evaluated_clip_instances") or row["clip_count"])
        if runtime.get("evaluated_clip_instances") or row.get("clip_count")
        else None
    )
    return (
        f"{_tex(_runtime_profile_label(row))} & "
        f"{_number(requests_per_clip, 1)} & "
        f"{_number(runtime['wall_seconds_per_clip'], 2)} & "
        f"{_number(runtime['request_e2e_seconds_p95'], 2)} & "
        f"{_number(runtime['prefill_seconds_per_clip'], 2)} & "
        f"{_number(runtime['decode_seconds_per_clip'], 2)} & "
        f"{_pct(runtime['mm_cache_hit_rate'])} & "
        f"{_pct(runtime['prefix_cache_token_hit_rate'])} & "
        f"{int(runtime.get('preemptions') or 0)} & "
        f"{_number(runtime['estimated_usd_per_1000_clips'], 2)} \\\\"
    )


def _road_serving_findings_text(
    profiles: list[dict[str, Any]],
) -> str:
    grouped = _find_smoke_profile(
        profiles,
        model="Nano",
        profile="road-adaptive-24-grouped",
        run_token="road-nano-development-smoke-v1",
    )
    nano_c1 = _find_smoke_profile(
        profiles,
        model="Nano",
        serving_profile="nano-server-default",
        concurrency=1,
        run_token="road-serving-road-only",
    )
    nano_c4 = _find_smoke_profile(
        profiles,
        model="Nano",
        serving_profile="nano-server-default",
        concurrency=4,
        run_token="road-serving-road-only",
    )
    nano_c8 = _find_smoke_profile(
        profiles,
        model="Nano",
        serving_profile="nano-server-default",
        concurrency=8,
        run_token="road-serving-road-only",
    )
    chunk_2048 = _find_smoke_profile(
        profiles,
        model="Nano",
        serving_profile="nano-chunked-2048",
        concurrency=4,
    )
    chunk_8192 = _find_smoke_profile(
        profiles,
        model="Nano",
        serving_profile="nano-chunked-8192",
        concurrency=4,
    )
    super_c4 = _find_smoke_profile(
        profiles,
        model="Super",
        serving_profile="super-weights-chunked-8192",
        concurrency=4,
    )
    super_c8 = _find_smoke_profile(
        profiles,
        model="Super",
        serving_profile="super-weights-chunked-8192",
        concurrency=8,
    )
    required = (
        grouped,
        nano_c1,
        nano_c4,
        nano_c8,
        chunk_2048,
        chunk_8192,
        super_c4,
        super_c8,
    )
    if any(row is None for row in required):
        return (
            "ROAD controlled sweepの全条件が揃っていないため、"
            "本稿では表に存在する実測値のみを報告する。"
        )

    assert grouped is not None
    assert nano_c1 is not None
    assert nano_c4 is not None
    assert nano_c8 is not None
    assert chunk_2048 is not None
    assert chunk_8192 is not None
    assert super_c4 is not None
    assert super_c8 is not None
    c1_runtime = nano_c1["runtime"]
    c4_runtime = nano_c4["runtime"]
    c8_runtime = nano_c8["runtime"]
    chunk2_runtime = chunk_2048["runtime"]
    chunk8_runtime = chunk_8192["runtime"]
    grouped_runtime = grouped["runtime"]
    super4_runtime = super_c4["runtime"]
    super8_runtime = super_c8["runtime"]
    batching_speedup = (
        float(c1_runtime["wall_seconds_per_clip"])
        / float(c4_runtime["wall_seconds_per_clip"])
    )
    batching_cost_reduction = 1.0 - (
        float(c4_runtime["estimated_usd_per_1000_clips"])
        / float(c1_runtime["estimated_usd_per_1000_clips"])
    )
    chunk_speedup = (
        float(chunk8_runtime["wall_seconds_per_clip"])
        / float(chunk2_runtime["wall_seconds_per_clip"])
    )
    return (
        "ROADの同一15 clipsをcold start後に処理したNanoでは、server defaultの"
        f"concurrency 1から4でthroughputが{batching_speedup:.2f}倍となり、"
        f"EC2費用は\\${_number(c1_runtime['estimated_usd_per_1000_clips'], 2)}から"
        f"\\${_number(c4_runtime['estimated_usd_per_1000_clips'], 2)}/1,000 clipsへ"
        f"{_pct_text(batching_cost_reduction)}低下した。一方、P95 latencyは"
        f"{_number(c1_runtime['request_e2e_seconds_p95'], 2)}秒から"
        f"{_number(c4_runtime['request_e2e_seconds_p95'], 2)}秒へ増加し、c8では"
        f"{_number(c8_runtime['request_e2e_seconds_p95'], 2)}秒となった。"
        f"concurrency 4におけるChunked Prefill 2,048 tokensは8,192 tokens比で"
        f"throughputが{chunk_speedup:.2f}倍であり、prefillは"
        f"{_number(chunk2_runtime['prefill_seconds_per_clip'], 2)}秒/clip対"
        f"{_number(chunk8_runtime['prefill_seconds_per_clip'], 2)}秒/clipであった。"
        f"ただしserver default c4の費用は"
        f"\\${_number(c4_runtime['estimated_usd_per_1000_clips'], 2)}/1,000 clipsで、"
        f"2,048-token条件の"
        f"\\${_number(chunk2_runtime['estimated_usd_per_1000_clips'], 2)}を下回り、"
        "Chunked Prefillの明確な優位は認められなかった。"
        "3-group requestでは同一mediaを連続再利用し、MM cache hit "
        f"{_pct_text(grouped_runtime['mm_cache_hit_rate'])}、Prefix cache hit "
        f"{_pct_text(grouped_runtime['prefix_cache_token_hit_rate'])}を観測した。"
        "Superではc4のpreemptionが"
        f"{int(super4_runtime.get('preemptions') or 0)}件であったのに対し、c8は"
        f"{int(super8_runtime.get('preemptions') or 0)}件、P95 "
        f"{_number(super8_runtime['request_e2e_seconds_p95'], 2)}秒となったため、"
        "Full Runにはc4を採用した。これらの結果は、throughput最大化とtail latency、"
        "KV-cache容量および再計算回避を同時に考慮する必要を示す。"
    )


def _find_smoke_profile(
    profiles: list[dict[str, Any]],
    *,
    model: str | None = None,
    profile: str | None = None,
    serving_profile: str | None = None,
    concurrency: int | None = None,
    run_token: str | None = None,
) -> dict[str, Any] | None:
    for row in profiles:
        if model and not str(row.get("model", "")).endswith(model):
            continue
        if profile and row.get("profile") != profile:
            continue
        if serving_profile and row.get("serving_profile") != serving_profile:
            continue
        if concurrency is not None and row.get("concurrency") != concurrency:
            continue
        if run_token and run_token not in str(row.get("run_id", "")):
            continue
        return row
    return None


def _supplemental_results_tex(
    error_analysis: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    serving_repeat: dict[str, Any] | None,
    cascade: dict[str, Any] | None,
) -> str:
    blocks: list[str] = []
    if sampling_audit:
        rows = "\n".join(
            (
                f"{_tex(_short_profile(row['profile']))} & "
                f"{float(row['mean_selected_frames']):.1f} & "
                f"{_pct(row['any_event_hit_rate'])} & "
                f"{_pct(row['two_event_frame_rate'])} & "
                f"{_pct(row['event_span_at_least_0_5s_rate'])} & "
                f"{_pct(row['both_boundary_contexts_rate'])} \\\\"
            )
            for row in sampling_audit["profiles"]
        )
        blocks.append(
            r"""\subsection{Sampling coverage}
GT event区間と入力timestampを照合し、eventを1枚以上含む率、2枚以上含む率、
0.5秒以上の時間span、およびevent前後contextを測定した。これは方式選定後の
品質評価とは別の原因診断であり、GT時刻を推論入力の選択には使用していない。
\begin{center}
\captionof{table}{Developmentにおけるsampling coverage}
\label{tab:sampling-coverage}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
条件 & frames & any & 2 frames & span & context\\
\midrule
"""
            + rows
            + r"""
\bottomrule
\end{tabular}}
\end{center}"""
        )
    if error_analysis:
        rows = "\n".join(
            (
                f"{_tex(str(row['model']).split('/')[-1])} & "
                f"{int(row['false_negative_count'])} & "
                f"{int(row['temporal_sampling_miss'])} & "
                f"{int(row['temporally_sparse_miss'])} & "
                f"{int(row['temporally_covered_miss'])} & "
                f"{_pct(row['sampling_or_sparse_fraction'])} \\\\"
            )
            for row in error_analysis["profiles"]
        )
        fractions = [
            float(row["sampling_or_sparse_fraction"])
            for row in error_analysis["profiles"]
        ]
        blocks.append(
            r"""\subsection{False negativeの原因分解}
GT陽性に対するfalse negativeを、event区間内frameが0枚のtemporal miss、
2枚未満または0.5秒未満のsparse coverage、および十分な時間coverageがある
covered missへ分類した。
\begin{center}
\captionof{table}{Evaluation Full Runのfalse negative原因}
\label{tab:error-analysis}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Model & FN & miss & sparse & covered & miss+sparse\\
\midrule
"""
            + rows
            + r"""
\bottomrule
\end{tabular}}
\end{center}
sampling missまたはsparse coverageで説明できるFNは"""
            + f"{_pct_text(min(fractions))}--{_pct_text(max(fractions))}"
            + r"""に留まった。covered missは、対象が小さい、遮蔽・夜間、
taxonomy境界、prompt表現、またはモデル能力を含むため、純粋なモデル誤りとは
断定しない。"""
        )
    if serving_repeat:
        rows = []
        for row in serving_repeat["groups"]:
            metrics = row["metrics"]
            label = _serving_repeat_label(row)
            rows.append(
                f"{_tex(label)} & "
                f"{int(row['replicate_count'])} & "
                f"{_mean_sd(metrics['wall_seconds_per_clip'], 2)} & "
                f"{_mean_sd(metrics['request_e2e_seconds_p95'], 2)} & "
                f"{_mean_sd(metrics['prefill_seconds_per_clip'], 2)} & "
                f"{_mean_sd(metrics['decode_seconds_per_clip'], 2)} & "
                f"{_mean_sd(metrics['estimated_usd_per_1000_clips'], 3)} "
                "\\\\"
            )
        blocks.append(
            r"""\end{multicols}
\begin{center}
\captionof{table}{ROAD 15 clipsを用いたNano serving 3反復（平均$\pm$標本標準偏差）}
\label{tab:serving-repeat}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
条件 & runs & wall/clip[s] & P95[s] & prefill[s] & decode[s] & USD/1k\\
\midrule
"""
            + "\n".join(rows)
            + r"""
\bottomrule
\end{tabular}}
\end{center}
\begin{multicols}{2}"""
        )
    if cascade:
        policy_labels = {
            "nano_only": "Nanoのみ",
            "nano_positive_count_at_least_3": "positive 3個以上をSuper",
            "nano_pedestrian_positive": "歩行者positiveをSuper",
            "nano_complex_temporal_positive": "複雑時系列positiveをSuper",
            "super_all": "Super全件",
        }
        rows = "\n".join(
            (
                f"{_tex(policy_labels.get(row['policy'], row['policy']))} & "
                f"{_pct(row['routing_fraction'])} & "
                f"{_pct(row['metrics']['precision'])} & "
                f"{_pct(row['metrics']['strict_recall'])} & "
                f"{_pct(row['metrics']['strict_f1'])} & "
                f"{_number(row['metrics']['selective_mcc'])} & "
                f"{_number(row['estimated_usd_per_1000_clips'], 2)} \\\\"
            )
            for row in cascade["policies"]
        )
        blocks.append(
            r"""\end{multicols}
\begin{center}
\captionof{table}{Nano予測だけでroutingするNano--Super cascade}
\label{tab:cascade}
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
方策 & Super率 & P & R & F1 & MCC & USD/1k\\
\midrule
"""
            + rows
            + r"""
\bottomrule
\end{tabular}
\end{center}
\begin{multicols}{2}
routing判定にGTは用いず、Nanoの予測positiveだけを使用した。したがって、
全件SuperとNanoのみの間に、実測された品質・費用の中間運用点を構成できる。"""
        )
    return "\n\n".join(blocks)


def _mean_sd(metric: dict[str, Any], digits: int) -> str:
    return (
        f"{float(metric['mean']):.{digits}f}"
        f"$\\pm${float(metric['sample_standard_deviation']):.{digits}f}"
    )


def _serving_repeat_label(row: dict[str, Any]) -> str:
    labels = {
        "nano-server-default": "server default",
        "nano-chunked-2048": "Chunked Prefill 2,048",
        "nano-chunked-8192": "Chunked Prefill 8,192",
    }
    return (
        f"{labels.get(row['serving_profile'], row['serving_profile'])} "
        f"(c{int(row['concurrency'])})"
    )


def _distribution_row(row: dict[str, Any]) -> str:
    distribution = row["distribution"]
    return (
        f"{_tex(_profile_label(row))} & "
        f"{_number(distribution['prevalence_mae'], 3)} & "
        f"{_number(distribution['label_distribution_js_divergence'], 3)} "
        "\\\\"
    )


def _tag_comparison_rows(profiles: list[dict[str, Any]]) -> str:
    tag_maps = {
        row["benchmark_key"]: {item["tag"]: item for item in row["tags"]}
        for row in profiles
    }
    tags = [item["tag"] for item in profiles[0]["tags"]]
    rows = []
    for tag in tags:
        reference = tag_maps[profiles[0]["benchmark_key"]][tag]
        cells = [
            _pct(tag_maps[row["benchmark_key"]][tag]["strict_f1"])
            for row in profiles
        ]
        positives = (
            int(reference["true_positive"])
            + int(reference["false_negative"])
            + int(reference["abstained_positive"])
        )
        rows.append(
            f"{_tex(TAG_LABELS.get(tag, tag))} & {positives} & "
            + " & ".join(cells)
            + " \\\\"
        )
    return "\n".join(rows)


def _curation_rows(profiles: list[dict[str, Any]]) -> str:
    available = [
        row for row in profiles if row.get("rare_scene_retrieval")
    ]
    if not available:
        return (
            "\\noindent 希少scene retrievalは、該当するrare tagまたは"
            "有効なrankingがないため算出しなかった。"
        )
    rows = []
    for profile in available:
        for budget in profile["rare_scene_retrieval"].get("budgets", []):
            rows.append(
                f"{_tex(_profile_label(profile))} & "
                f"{_pct(budget.get('review_fraction'))} & "
                f"{_pct(budget.get('precision_mean'))} & "
                f"{_pct(budget.get('recall_mean'))} & "
                f"{_number(budget.get('enrichment_mean'), 2)} \\\\"
            )
    if not rows:
        return "\\noindent 希少scene retrievalは有効なbudgetがなく算出しなかった。"
    return r"""\begin{center}
\centering
\captionof{table}{予測positiveを優先した希少scene人手確認}
\label{tab:curation}
\scriptsize
\begin{tabular}{lrrrr}
\toprule
条件 & 確認率 & P@K & R@K & enrichment\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}"""


def _algorithm_smoke_profiles(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred = [
        "road-uniform-1fps-ts",
        "road-uniform-2fps-ts",
        "road-uniform-4fps-ts",
        "road-uniform-8fps-ts",
        "road-adaptive-24-ts",
        "road-adaptive-24-reasoned",
        "road-adaptive-24-grouped",
    ]
    priority = {name: index for index, name in enumerate(preferred)}
    selected = [
        row
        for row in profiles
        if str(row.get("profile")) in priority
        and "road-nano-development-smoke-v1"
        in str(row.get("run_id", ""))
    ]
    if not selected:
        return []
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        unique[str(row.get("profile", "profile"))] = row
    return sorted(
        unique.values(),
        key=lambda row: priority[str(row.get("profile"))],
    )


def _serving_sweep_profiles(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in profiles
        if (
            "road-serving-road-only"
            in str(row.get("run_id", ""))
            or str(row.get("run_id", "")).startswith(
                "road-super-reasoned-c"
            )
        )
        and row.get("profile") == "road-adaptive-24-reasoned"
    ]
    return sorted(
        selected,
        key=lambda row: (
            0 if str(row.get("model", "")).endswith("Nano") else 1,
            _serving_profile_order(str(row.get("serving_profile") or "")),
            int(row.get("concurrency") or 0),
        ),
    )


def _serving_profile_order(profile: str) -> int:
    order = {
        "nano-server-default": 0,
        "nano-chunked-2048": 1,
        "nano-chunked-8192": 2,
        "super-weights-chunked-8192": 3,
    }
    return order.get(profile, 99)


def _runtime_profile_label(row: dict[str, Any]) -> str:
    run_id = str(row.get("run_id", ""))
    serving_profile = str(row.get("serving_profile") or "")
    if (
        "road-serving-road-only" in run_id
        or run_id.startswith("road-super-reasoned-c")
    ):
        model = str(row.get("model", "model")).split("/")[-1]
        serving_labels = {
            "nano-server-default": "server default",
            "nano-chunked-2048": "Chunked Prefill 2,048",
            "nano-chunked-8192": "Chunked Prefill 8,192",
            "super-weights-chunked-8192": "TP4, Chunked Prefill 8,192",
        }
        label = f"{model} / {serving_labels.get(serving_profile, serving_profile)}"
    else:
        label = _profile_label(row)
    concurrency = row.get("concurrency")
    return (
        f"{label} (c{int(concurrency)})"
        if concurrency is not None
        else label
    )


def _model_comparison_text(
    profiles: list[dict[str, Any]],
) -> dict[str, str]:
    compared = next(
        (
            row
            for row in profiles
            if row.get("comparison_reference")
            and row.get("paired_delta_ci95", {}).get("strict_f1")
        ),
        None,
    )
    if compared is None:
        return {
            "abstract": "",
            "discussion": (
                "モデル間のpaired比較は、同一clipを完走した2条件が"
                "揃わなかったため算出していない。"
            ),
        }
    reference = next(
        (
            row
            for row in profiles
            if row.get("benchmark_key")
            == compared.get("comparison_reference")
        ),
        None,
    )
    if reference is None:
        return {"abstract": "", "discussion": ""}
    delta = (
        float(compared["micro"]["strict_f1"])
        - float(reference["micro"]["strict_f1"])
    )
    ci = compared["paired_delta_ci95"]["strict_f1"]
    reference_cost = reference["runtime"].get(
        "estimated_usd_per_1000_clips"
    )
    compared_cost = compared["runtime"].get(
        "estimated_usd_per_1000_clips"
    )
    cost_ratio = (
        float(compared_cost) / float(reference_cost)
        if reference_cost and compared_cost is not None
        else None
    )
    candidate = _tex(_profile_label(compared))
    baseline = _tex(_profile_label(reference))
    sign = "+" if delta >= 0 else ""
    delta_text = (
        f"{sign}{100 * delta:.1f} points"
        f"（95\\% CI: {100 * float(ci[0]):+.1f}--"
        f"{100 * float(ci[1]):+.1f}）"
    )
    ratio_text = (
        f"、定常推論費用は{cost_ratio:.1f}倍" if cost_ratio is not None else ""
    )
    if delta > 0:
        deployment = (
            "品質を優先する場合はcandidateを用い、費用を優先する大量一次"
            "ラベリングにはbaselineを用いる二段構成が妥当である。"
        )
    else:
        deployment = (
            "candidateは本条件で品質改善を示さず、追加費用を正当化できないため、"
            "baselineを既定とすべきである。"
        )
    return {
        "abstract": (
            f"{baseline}に対する{candidate}のpaired F1差は"
            f"{delta_text}であった。"
        ),
        "discussion": (
            f"{baseline}に対する{candidate}のpaired F1差は"
            f"{delta_text}{ratio_text}であった。"
            + deployment.replace("candidate", candidate).replace(
                "baseline", baseline
            )
        ),
    }


def _cost_projection_text(profiles: list[dict[str, Any]]) -> str:
    nano = next(
        (
            row
            for row in profiles
            if str(row.get("model", "")).endswith("Nano")
        ),
        None,
    )
    super_profile = next(
        (
            row
            for row in profiles
            if str(row.get("model", "")).endswith("Super")
        ),
        None,
    )
    if nano is None or super_profile is None:
        return "Nano/Superの両Full Runがないため費用感度を算出しなかった。"
    nano_cost = float(
        nano["runtime"]["estimated_usd_per_1000_clips"]
    )
    super_cost = float(
        super_profile["runtime"]["estimated_usd_per_1000_clips"]
    )
    review_fraction = 0.05
    hybrid_cost = nano_cost + review_fraction * super_cost
    reduction = 1.0 - hybrid_cost / super_cost
    return (
        f"定常推論だけを線形換算すると、100万clipsのEC2費用はNano "
        f"\\${_number(nano_cost * 1000, 0)}、Super "
        f"\\${_number(super_cost * 1000, 0)}である。全件をNanoで処理し、"
        f"予測positive等で選んだ上位5\\%だけをSuperで再判定する場合、"
        f"費用は\\${_number(hybrid_cost, 2)}/1,000 clips、"
        f"100万clipsで\\${_number(hybrid_cost * 1000, 0)}となり、"
        f"Super全件適用比で{_pct_text(reduction)}低い。これは費用の感度分析であり、"
        "二段構成の最終F1を直接測定した結果ではない。model download、compile、"
        "idle、storage、network費用を含まないため、低稼働率では実費が増える。"
    )


def _tag_findings_text(profile: dict[str, Any]) -> str:
    rows = [
        row
        for row in profile.get("tags", [])
        if row.get("strict_f1") is not None
    ]
    if not rows:
        return ""
    ranked = sorted(rows, key=lambda row: float(row["strict_f1"]))
    weak = ranked[:3]
    strong = list(reversed(ranked[-3:]))

    def describe(row: dict[str, Any]) -> str:
        positives = (
            int(row.get("true_positive", 0))
            + int(row.get("false_negative", 0))
            + int(row.get("abstained_positive", 0))
        )
        return (
            f"{_tex(TAG_LABELS.get(row['tag'], row['tag']))}"
            f"（F1 {_pct(row['strict_f1'])}, positive $n={positives}$）"
        )

    return (
        "タグ別では、"
        + "、".join(describe(row) for row in strong)
        + "が高かった一方、"
        + "、".join(describe(row) for row in weak)
        + "が低く、希少maneuverと歩行者の意図・運動状態には"
        "追加のtrack情報または局所高FPS evidenceが必要である。"
    )


def _profile_label(row: dict[str, Any]) -> str:
    model = str(row.get("model", "model")).split("/")[-1]
    return f"{model} / {_short_profile(str(row.get('profile', 'profile')))}"


def _short_profile(profile: str) -> str:
    return (
        profile.replace("road-", "")
        .replace("uniform-", "uniform ")
        .replace("adaptive-", "adaptive ")
        .replace("-ts", "+timestamp")
    )


def _render_markdown(
    evaluation: dict[str, Any],
    development: dict[str, Any],
    smoke: dict[str, Any],
    manifest: dict[str, Any],
    audit: dict[str, Any],
    error_analysis: dict[str, Any] | None = None,
    sampling_audit: dict[str, Any] | None = None,
    serving_repeat: dict[str, Any] | None = None,
    cascade: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# 公開自動運転動画に対する時系列シーンラベリング評価",
        "",
        "本成果物の品質・性能・費用評価は、すべて公開ROAD dataset上で実施した。",
        "",
        "## Dataset",
        "",
        f"- Annotated frames: {audit['annotated_frame_count']:,}",
        f"- Source videos: {manifest['source_video_count']}",
        f"- Development clips: {manifest['split_clip_counts']['development']}",
        f"- Evaluation clips: {manifest['split_clip_counts']['evaluation']}",
        f"- Tags: {len(manifest['tags'])}",
        "",
        "## Evaluation Full Run",
        "",
        "| Model / profile | Clips | Precision | Recall | F1 | Accuracy | "
        "Balanced Accuracy | MCC | USD/1k clips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["profiles"]:
        micro = row["micro"]
        lines.append(
            f"| {_profile_label(row)} | {row['clip_count']} | "
            f"{_pct(micro['precision'])} | "
            f"{_pct(micro['strict_recall'])} | "
            f"{_pct(micro['strict_f1'])} | "
            f"{_pct(micro['overall_accuracy'])} | "
            f"{_pct(micro['strict_balanced_accuracy'])} | "
            f"{_number(micro['selective_mcc'])} | "
            f"{_number(row['runtime']['estimated_usd_per_1000_clips'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Development Ablation",
            "",
            "| Profile | Precision | Recall | F1 | Balanced Accuracy | MCC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in development["profiles"]:
        micro = row["micro"]
        lines.append(
            f"| {row['profile']} | {_pct(micro['precision'])} | "
            f"{_pct(micro['strict_recall'])} | "
            f"{_pct(micro['strict_f1'])} | "
            f"{_pct(micro['strict_balanced_accuracy'])} | "
            f"{_number(micro['selective_mcc'])} |"
        )
    lines.extend(
        [
            "",
            "## ROAD Serving Controlled Sweep",
            "",
            "| Model / condition | c | Wall/clip | P95 | Prefill/clip | "
            "Decode/clip | MM cache | Prefix cache | Preemptions | USD/1k |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _serving_sweep_profiles(smoke["profiles"]):
        runtime = row["runtime"]
        lines.append(
            f"| {_runtime_profile_label(row)} | {row['concurrency']} | "
            f"{_number(runtime['wall_seconds_per_clip'], 2)} | "
            f"{_number(runtime['request_e2e_seconds_p95'], 2)} | "
            f"{_number(runtime['prefill_seconds_per_clip'], 2)} | "
            f"{_number(runtime['decode_seconds_per_clip'], 2)} | "
            f"{_pct(runtime['mm_cache_hit_rate'])} | "
            f"{_pct(runtime['prefix_cache_token_hit_rate'])} | "
            f"{int(runtime.get('preemptions') or 0)} | "
            f"{_number(runtime['estimated_usd_per_1000_clips'], 2)} |"
        )
    lines.extend(
        [
            "",
            _road_serving_findings_text(smoke["profiles"]).replace(
                "\\$", "$"
            ),
        ]
    )
    lines.extend(
        _supplemental_markdown(
            error_analysis,
            sampling_audit,
            serving_repeat,
            cascade,
        )
    )
    return "\n".join(lines) + "\n"


def _supplemental_markdown(
    error_analysis: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    serving_repeat: dict[str, Any] | None,
    cascade: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    if sampling_audit:
        lines.extend(
            [
                "",
                "## Sampling Coverage",
                "",
                "| Profile | Frames | Any hit | 2 frames | Span >=0.5s | Context |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sampling_audit["profiles"]:
            lines.append(
                f"| {row['profile']} | {row['mean_selected_frames']:.1f} | "
                f"{_pct(row['any_event_hit_rate'])} | "
                f"{_pct(row['two_event_frame_rate'])} | "
                f"{_pct(row['event_span_at_least_0_5s_rate'])} | "
                f"{_pct(row['both_boundary_contexts_rate'])} |"
            )
    if error_analysis:
        lines.extend(
            [
                "",
                "## False Negative Diagnosis",
                "",
                "| Model | FN | Temporal miss | Sparse | Covered | Sampling/sparse |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in error_analysis["profiles"]:
            lines.append(
                f"| {str(row['model']).split('/')[-1]} | "
                f"{row['false_negative_count']} | "
                f"{row['temporal_sampling_miss']} | "
                f"{row['temporally_sparse_miss']} | "
                f"{row['temporally_covered_miss']} | "
                f"{_pct(row['sampling_or_sparse_fraction'])} |"
            )
    if serving_repeat:
        lines.extend(
            [
                "",
                "## Repeated Serving Sweep",
                "",
                "| Condition | Runs | Wall/clip mean | P95 mean | USD/1k mean |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in serving_repeat["groups"]:
            metrics = row["metrics"]
            lines.append(
                f"| {_serving_repeat_label(row)} | {row['replicate_count']} | "
                f"{metrics['wall_seconds_per_clip']['mean']:.3f} | "
                f"{metrics['request_e2e_seconds_p95']['mean']:.3f} | "
                f"{metrics['estimated_usd_per_1000_clips']['mean']:.3f} |"
            )
    if cascade:
        lines.extend(
            [
                "",
                "## Nano-Super Cascade",
                "",
                "| Policy | Routed | Precision | Recall | F1 | MCC | USD/1k |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in cascade["policies"]:
            lines.append(
                f"| {row['policy']} | {_pct(row['routing_fraction'])} | "
                f"{_pct(row['metrics']['precision'])} | "
                f"{_pct(row['metrics']['strict_recall'])} | "
                f"{_pct(row['metrics']['strict_f1'])} | "
                f"{_number(row['metrics']['selective_mcc'])} | "
                f"{_number(row['estimated_usd_per_1000_clips'], 2)} |"
            )
    return lines


def _render_reproduction(
    evaluation_summary_path: Path,
    development_summary_path: Path,
    smoke_summary_path: Path,
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    supplemental_paths: dict[str, Path] | None = None,
) -> str:
    def relative(path: Path) -> str:
        return str(path.relative_to(output_dir))

    supplemental_paths = supplemental_paths or {}
    supplemental_lines = "\n".join(
        f"- {name}: `{relative(path)}`"
        for name, path in supplemental_paths.items()
    )
    supplemental_args = "\n".join(
        {
            "error-analysis.json": (
                f"  --error-analysis <paper-dir>/{relative(path)} \\"
            ),
            "sampling-audit.json": (
                f"  --sampling-audit <paper-dir>/{relative(path)} \\"
            ),
            "serving-repeat-summary.json": (
                f"  --serving-repeat-summary <paper-dir>/{relative(path)} \\"
            ),
            "cascade-summary.json": (
                f"  --cascade-summary <paper-dir>/{relative(path)} \\"
            ),
        }[name]
        for name, path in supplemental_paths.items()
    )
    return f"""# Reproduction

The paper is generated only from locked ROAD artifacts and complete-GT
evaluation summaries.

## Inputs

- Evaluation summary: `{relative(evaluation_summary_path)}`
- Development summary: `{relative(development_summary_path)}`
- Serving smoke summary: `{relative(smoke_summary_path)}`
- ROAD GT manifest: `{relative(manifest_path)}`
- ROAD annotation audit: `{relative(audit_path)}`
{supplemental_lines}

## Generate the paper

```bash
cd benchmark
PYTHONPATH=. .venv/bin/python -m benchmark_tool.cli road-report \\
  --evaluation-summary <paper-dir>/{relative(evaluation_summary_path)} \\
  --development-summary <paper-dir>/{relative(development_summary_path)} \\
  --smoke-summary <paper-dir>/{relative(smoke_summary_path)} \\
  --manifest <paper-dir>/{relative(manifest_path)} \\
  --audit <paper-dir>/{relative(audit_path)} \\
{supplemental_args}
  --output-dir <output-directory>
```

The artifact manifest records SHA-256 digests for every input and output.
"""


def _pct(value: Any) -> str:
    if value is None:
        return "--"
    return f"{100 * float(value):.1f}"


def _pct_text(value: Any) -> str:
    value_text = _pct(value)
    return value_text if value_text == "--" else f"{value_text}\\%"


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _tex(value: Any) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(source, target)
    return text
