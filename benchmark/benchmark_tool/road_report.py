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
    motion_prior_summary_path: Path | None = None,
    composed_motion_summary_path: Path | None = None,
    context_fusion_summary_path: Path | None = None,
    opr_motion_lock_path: Path | None = None,
    visual_motion_lock_path: Path | None = None,
    map_context_lock_path: Path | None = None,
    prompt_development_summary_path: Path | None = None,
    prompt_selection_lock_path: Path | None = None,
    prompt_audit_path: Path | None = None,
    recall_diagnostic_summary_path: Path | None = None,
    recall_confidence_baseline_path: Path | None = None,
    recall_confidence_oracle_path: Path | None = None,
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
    motion_prior = (
        load_json(motion_prior_summary_path)
        if motion_prior_summary_path
        else None
    )
    composed_motion = (
        load_json(composed_motion_summary_path)
        if composed_motion_summary_path
        else None
    )
    context_fusion = (
        load_json(context_fusion_summary_path)
        if context_fusion_summary_path
        else None
    )
    opr_motion = (
        load_json(opr_motion_lock_path) if opr_motion_lock_path else None
    )
    visual_motion = (
        load_json(visual_motion_lock_path)
        if visual_motion_lock_path
        else None
    )
    map_context = (
        load_json(map_context_lock_path)
        if map_context_lock_path
        else None
    )
    prompt_development = (
        load_json(prompt_development_summary_path)
        if prompt_development_summary_path
        else None
    )
    prompt_selection = (
        load_json(prompt_selection_lock_path)
        if prompt_selection_lock_path
        else None
    )
    prompt_audit = (
        load_json(prompt_audit_path) if prompt_audit_path else None
    )
    recall_diagnostic = (
        load_json(recall_diagnostic_summary_path)
        if recall_diagnostic_summary_path
        else None
    )
    recall_confidence_baseline = (
        load_json(recall_confidence_baseline_path)
        if recall_confidence_baseline_path
        else None
    )
    recall_confidence_oracle = (
        load_json(recall_confidence_oracle_path)
        if recall_confidence_oracle_path
        else None
    )
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
        "motion-prior-summary.json": motion_prior_summary_path,
        "composed-motion-summary.json": composed_motion_summary_path,
        "context-fusion-summary.json": context_fusion_summary_path,
        "opr-motion-lock.json": opr_motion_lock_path,
        "visual-motion-lock.json": visual_motion_lock_path,
        "map-context-lock.json": map_context_lock_path,
        "prompt-development-summary.json": prompt_development_summary_path,
        "prompt-selection-lock.json": prompt_selection_lock_path,
        "prompt-audit.json": prompt_audit_path,
        "recall-diagnostic-summary.json": recall_diagnostic_summary_path,
        "recall-confidence-baseline.json": recall_confidence_baseline_path,
        "recall-confidence-oracle.json": recall_confidence_oracle_path,
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
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        bundled_inputs[name] = destination

    tex_path = output_dir / "ROAD_COSMOS3_SCENE_LABELING_PAPER.tex"
    markdown_path = output_dir / "ROAD_COSMOS3_SCENE_LABELING_PAPER.md"
    reproduction_path = output_dir / "REPRODUCTION.md"
    review_path = output_dir / "ADVERSARIAL_SELF_REVIEW.md"
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
            motion_prior,
            composed_motion,
            context_fusion,
            opr_motion,
            visual_motion,
            map_context,
            prompt_development,
            prompt_selection,
            prompt_audit,
            recall_diagnostic,
            recall_confidence_baseline,
            recall_confidence_oracle,
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
            motion_prior,
            composed_motion,
            context_fusion,
            opr_motion,
            visual_motion,
            map_context,
            prompt_development,
            prompt_selection,
            prompt_audit,
            recall_diagnostic,
            recall_confidence_baseline,
            recall_confidence_oracle,
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
    review_path.write_text(_render_adversarial_self_review(), encoding="utf-8")

    pdf_path: Path | None = None
    if compile_pdf:
        tectonic = shutil.which("tectonic")
        if not tectonic:
            raise FileNotFoundError("tectonic is required to compile the PDF")
        subprocess.run(
            [
                tectonic,
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
        review_path,
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
                    "sha256": sha256_file(bundled_inputs[name]),
                }
                for name in source_inputs
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
    motion_prior: dict[str, Any] | None = None,
    composed_motion: dict[str, Any] | None = None,
    context_fusion: dict[str, Any] | None = None,
    opr_motion: dict[str, Any] | None = None,
    visual_motion: dict[str, Any] | None = None,
    map_context: dict[str, Any] | None = None,
    prompt_development: dict[str, Any] | None = None,
    prompt_selection: dict[str, Any] | None = None,
    prompt_audit: dict[str, Any] | None = None,
    recall_diagnostic: dict[str, Any] | None = None,
    recall_confidence_baseline: dict[str, Any] | None = None,
    recall_confidence_oracle: dict[str, Any] | None = None,
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
    best_cost_value = best_eval["runtime"][
        "estimated_usd_per_1000_clips"
    ]
    best_cost = _number(best_cost_value, 2)
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
    motion_text = _public_motion_summary_text(
        motion_prior,
        composed_motion,
        baseline_cost=best_cost_value,
    )
    motion_method = _public_motion_method_tex(
        motion_prior,
        opr_motion,
        visual_motion,
    )
    if motion_text["metrics"] is not None:
        motion_metrics = motion_text["metrics"]
        assert isinstance(motion_metrics, dict)
        if float(motion_metrics["strict_f1"]) > float(
            best_eval["micro"]["strict_f1"]
        ):
            best_name = str(motion_text["name"])
            best_f1 = _pct(motion_metrics["strict_f1"])
            best_mcc = _number(motion_metrics["selective_mcc"])
            best_cost = _number(motion_text["cost"], 2)
            f1_ci_text = str(motion_text["f1_ci"])
    supplemental_results = _supplemental_results_tex(
        error_analysis,
        sampling_audit,
        serving_repeat,
        cascade,
        motion_prior,
        composed_motion,
        context_fusion,
        opr_motion,
        visual_motion,
        map_context,
    )
    prompt_method = _prompt_reasoning_method_tex(
        prompt_selection,
        prompt_audit,
    )
    prompt_results = _prompt_reasoning_results_tex(
        prompt_development,
        prompt_selection,
        prompt_audit,
    )
    prompt_summary = _prompt_reasoning_summary_text(
        prompt_development,
        prompt_selection,
    )
    recall_method = _selective_recall_method_tex(recall_diagnostic)
    recall_results = _selective_recall_results_tex(
        recall_diagnostic,
        recall_confidence_baseline,
        recall_confidence_oracle,
    )
    recall_summary = _selective_recall_summary_text(
        recall_diagnostic,
        recall_confidence_oracle,
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
方式探索はclass-balanced Smoke、方式固定はdevelopment、最終性能推定はevaluationへ分け、
複雑な候補は単純baselineを上回った場合だけ次段階へ進めた。
Evaluationで母集団性能まで確認した構成は、timestamp付き24-frame evidence、
Super、および固定VO priorである。これに対し、タグ限定Hybrid Core Reasoningは
developmentで選定し、候補駆動48-frame Recall passはGTを用いた難例診断で上限を調べた。
従って後二者は提案系のextensionであるが、Evaluation性能へ数値合成しない。
evaluationは@@EVAL_CLIPS@@ clips、@@EVAL_VIDEOS@@ source videos、
@@KNOWN_PAIRS@@ known clip--tag pairsを含む。最良条件@@BEST_NAME@@は
micro F1 @@BEST_F1@@（source-video block bootstrap 95\% CI:
@@F1_CI@@）、MCC @@BEST_MCC@@、推定費用
\$@@BEST_COST@@/1,000 clipsであった。@@MODEL_COMPARISON@@
@@MOTION_ABSTRACT@@
@@PROMPT_ABSTRACT@@
@@RECALL_ABSTRACT@@
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
実用的価値を測ることである。主な貢献は次の7点である。
\begin{enumerate}
  \item ROADの密な注釈から、unknownを明示した再現可能な13タグ・8秒clip benchmarkを構築した。
  \item 24-frame budget下でUniform、Adaptive、Hybridおよび診断用samplingを比較し、
        timestamp manifestとreasoned promptを組み合わせたevidence生成法を設計した。
  \item タグの意味、肯定証拠、時系列検査、除外条件を分離し、難しい5タグだけを
        詳細化するHybrid Core Reasoningをdevelopmentで固定した。
  \item 低Recallタグについて、GT時刻を用いる24/48-frame Oracle、空間crop、
        匿名trackおよびタグ別YES/NO scoreを段階的に比較し、見逃し原因を分解した。
  \item continuous batching、Chunked Prefill、MM/prefix cache、request順序を統制し、
        Prefill/Decode、tail latencyおよびGPU費用への作用を分離した。
  \item Nano--Super cascade、分布誤差、希少scene retrieval、source-video block bootstrapを
        含む、品質と運用費用を同時評価する再現可能な実験系を提示した。
  \item 公開RTK派生軌跡とvideo-derived VOをquality gate付きで統合し、
        evaluationへ再調整せず適用できる自車運動fusionを構築した。
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

\section{問題設定と評価方針}
本研究は、8秒の走行clip $v\in\mathcal{V}$に対し、複数の道路事象タグ
$t\in\mathcal{T}$を同時付与するmulti-label classificationとして定式化する。
ROADの注釈からclip--tag pairごとにpositive、negative、unknownを構成し、unknownは
品質指標の母数から除外した。一方、モデルが構文解析可能な回答を返さない場合や、
要求したタグを欠落させた場合はabstention $\bot$として保存し、正答へ丸めずstrict指標へ
反映した。この扱いにより、出力失敗を陰性予測として隠蔽せず、推論系全体の信頼性を
分類品質と同時に評価する。

品質比較にはPrecision、Recall、F1、Accuracy、Balanced Accuracy（BA）および
Matthews Correlation Coefficient（MCC）を用いるが、モデル選定の中心はF1とMCCとした。
Accuracyは負例の多いmulti-label問題で過大に見え得るため補助指標とし、出力pair coverageが
99\%未満の条件は候補から除外した。また、近接するclipを独立標本と仮定すると信頼区間を
過小評価するため、個々のclipではなくsource videoを再標本化単位とする10,000回の
block bootstrapを採用した。

自動ラベルをdataset解析へ用いる妥当性は、タグ別prevalenceのMean Absolute Error（MAE）と
Jensen--Shannon divergence（JSD）で測定した。前者は各タグの出現率を何ポイント誤るか、
後者はタグ構成全体がGT分布からどの程度乖離するかを表す。希少scene抽出については、
予測positiveを先に人手確認した場合のPrecision@K、Recall@Kおよびrandom sampling比
enrichmentを報告し、分類精度とは別にデータキュレーション上の効用を評価した。

\subsection{研究課題と検証仮説}
本稿では、品質だけを最大化する単一のmodel比較ではなく、入力生成からGPU servingまでを
含む系全体を対象として、次の研究課題を設定した。
\begin{description}
  \item[RQ1:] 限られたframe budgetの下で、均等sampling、motion集中sampling、
        timestamp表現および判定手続のうち、どの要因が時系列タグの品質を支配するか。
        当初は短時間eventへ局所高FPSを割り当てるAdaptive/Hybrid方式が有利と仮定した。
  \item[RQ2:] タグ定義へ肯定証拠、時系列検査、除外条件を追加するReasoning強化は、
        全タグ一律よりも難しいタグへ限定した方が品質とPrompt長を両立できるか。
  \item[RQ3:] NanoからSuperへのmodel規模拡大は、見逃し削減と誤検出削減のどちらへ
        主に作用し、その改善は追加EC2費用を正当化するか。
  \item[RQ4:] continuous batching、Chunked Prefill、MM/prefix cacheおよびrequest分割は、
        throughput、tail latency、Prefill/Decode、費用へそれぞれどのように作用するか。
  \item[RQ5:] 映像だけでは曖昧な自車走行・停止・旋回を、公開軌跡またはvideo-derived
        Visual Odometry（VO）で補助すると品質を改善できるか。また、contextを再びVLMへ
        入力する生成的fusionと、タグ単位の決定論的fusionのどちらが安定するか。
  \item[RQ6:] event時刻を既に含むFalse Negativeは、局所48-frame化、空間crop、
        track情報、またはタグ別の未校正YES scoreのどの段階で回復し、どの誤りが残るか。
  \item[RQ7:] 生成ラベルは個別clipの分類に留まらず、dataset分布推定および希少sceneの
        人手確認順序付けへ利用できるか。
\end{description}
これらを同時に変更すると因果を解釈できないため、RQ1ではmodelとserving、RQ2では
mediaとmodel、RQ4ではmediaとprompt、RQ5では映像baselineを固定した。各候補はSmoke、development、
evaluationの順に段階的に絞り込み、evaluationは方式固定後の性能推定にのみ用いた。

\section{公開GTの構築}
ROAD train/validation 18動画の@@ANNOTATED_FRAMES@@ annotated frames
（全@@ANNOTATION_FRAMES@@ framesの@@ANNOTATED_RATIO@@）を使用した。
8秒window、8秒stride、12 FPS原時刻で1,044 clipsを生成し、ROAD公式fold 3に従って
development @@DEV_CLIPS@@ clips（3動画）とevaluation @@TEST_CLIPS@@ clips
（15動画）へ分離した。各タグは6 annotated frames以上、存在系は3 frames以上を
positiveとし、全frameが注釈済みで対象eventがないclipをnegative、境界欠損を
unknownとした。evaluationのunknownは@@UNKNOWN_PAIRS@@ pairsである。
使用タグは@@TAGS@@である。

\section{提案手法}
\subsection{全体構成}
提案系は、(1) 元動画から24-frameの基準evidenceを生成し、(2) 各frameの元動画時刻を
manifestとして保持し、(3) 難しいタグだけ詳細化したHybrid Core Reasoningで一次判定し、
(4) 必要な自車運動タグだけを公開VO priorで補正し、(5) 軽量detector/tracker等の
外部証拠と一次negativeが不一致な場合だけ候補化し、(6) event近傍48 framesとタグ別
YES/NO scoreでRecallを補い、(7) 用途に応じてNanoとSuperをroutingする、という
階層構成である。

clip $v$から選択したframe集合を$M(v)$、その実時刻列を$\tau(v)$、タグ定義と判定手続を
$D$、VLMを$f_\theta$とすると、映像予測は
\[
 \hat{\mathbf{y}}(v)=f_\theta\!\left(M(v),\tau(v),D\right)
\]
で表される。さらにVO特徴$\mathbf{z}(v)$とdevelopmentで固定したタグ別規則
$g_t(\cdot;\phi_t)$を用い、対象タグ$t$だけを
\[
 \tilde{y}_t(v)=g_t\!\left(\hat{y}_t(v),\mathbf{z}(v);\phi_t\right)
\]
へ置換する。規則を持たないタグは$\tilde{y}_t=\hat{y}_t$とし、物体・歩行者・信号の
予測を自車運動contextで不用意に変化させない。推論時にGTは使用せず、$\phi_t$は
developmentで一度だけ固定する。

本設計の要点は、VLMへ多量のcontextを与えて全タグを常時再生成させるのではなく、
映像が得意な外観・意味判定と、VOが得意な自車状態判定を責務分離する点にある。
また、48-frame passは全clipへ適用せず、一次判定と構造化証拠が矛盾するタグだけへ
限定する。GT event時刻を使った結果はこの条件付きpassの上限診断であり、本番の候補時刻は
detector/tracker、lane、VOから推定しなければならない。
また、Sampling、Prompt、Model、Serving、Fusionを独立した操作因子として扱い、
品質差がどの処理に由来するかを追跡可能にした。

\subsection{Evidence mediaの構成}
本稿でいうevidence mediaとは、元動画から選択した静止frameを時系列順に連結した、
VLM入力用の短い動画である。元動画をそのまま入力する方式と比べ、frame budgetを明示的に
管理でき、sampling、prompt、servingの各要因を独立に比較できる。8秒clip当たりの基準budgetを
24 framesとし、次の三方式を主に検討した。

\subsubsection*{Uniform sampling}
clip全域を一定間隔で走査する。最終候補では3 FPS相当の24 timestampsを選択した。
detectorやmotion scoreに依存しないため、意味的には重要だが画素変化の小さい停止、
信号状態、遠方物体を一様に観測できる。また、処理内容が単純で、動画ごとの前処理時間と
選択frame数を予測しやすい。

\subsubsection*{Adaptive sampling}
元動画を4 FPSで軽量解析し、frame差分とoptical flowからmotion peakを求めた。近接peakは
2秒のnon-maximum suppressionで統合し、最大2箇所を採用する。全域を1 FPSで保持した上で、
peak前後2秒を2 FPS、中心前後1秒を4 FPSへ高密度化し、24-frame上限を超える場合は
優先度の低いframeと置換した。したがって、高FPS frameを無制限に追加する方式ではない。

\subsubsection*{Hybrid sampling}
全域を2 FPSで保持しつつ、8 FPSのmotion解析で選んだ1 peakの前後1秒を補強し、
中心前後0.5秒だけを8 FPS相当とする。Uniformの広域coverageとAdaptiveの局所密度を
同一24-frame budgetで両立させる設計である。

これらに加え、局所burstを32/48 framesへ増やす条件、GT eventの開始・中央・終了を
優先するdiagnostic oracle、道路上方・左右端を拡大するmulti-scale mosaicをdevelopment
限定で評価した。oracleはsamplingの理論上限を診断するための条件であり、方式選定や
evaluationには使用していない。multi-scaleも一部小物体を回復したがfalse positiveを
増加させたため、Full Run候補から除外した。

\subsection{実時間を保持するtimestamp表現}
evidence mediaは4 FPSで再生されるため、動画内の見かけのframe間隔は元動画の実時間間隔と
一致しない。特にAdaptive samplingでは、隣接frameが0.125秒差の場合と数秒差の場合が混在する。
そこで、選択した全frameの元動画時刻をtext manifestとしてpromptへ付与した。これにより、
モデルは単なる画像順序だけでなく、停止の持続時間、接近・離反の速度、右左折前後の時間関係を
解釈できる。timestamp有無のみを比較する条件では、media bytesをhard linkで共有し、
入力画像差が結果へ混入しないようにした。

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
\subsubsection*{Timestamp-only prompt}
全13タグの定義とtimestamp manifestを提示し、固定順のbinary arrayを返させる。
構造化出力として単純である一方、各タグの時間的根拠を順に照合する手続を明示しないため、
後述の実験ではnegative側へ過度に保守的になる傾向を示した。

\subsubsection*{Reasoned prompt}
全frameを時系列順に確認し、タグごとの肯定条件と除外条件を照合してから、
最終的なpositive indexのみを出力する手続を指示した。自由記述の説明文は要求せず、
\texttt{FINAL\_POSITIVE\_INDICES}に相当する短い配列へ制約する。これにより、
判定前の確認手順をprompt上で規定しつつ、decode token数とparse failureを抑える。
最大出力は64 tokensであり、長い自然言語説明を生成する方式ではない。

@@PROMPT_METHOD@@

\subsubsection*{Grouped request}
13タグをego action、road-user action、object/signalの3群へ分割する。同一media bytes、
同一timestamp manifest、同一system prefixを維持し、群固有の判定基準だけを末尾へ加える。
タグ間干渉の低減が期待できる一方、1 clip当たり3 requestとなるため、品質改善が追加latencyを
正当化できるかを実測した。同一clipの3 requestは連続投入し、multimodal cacheとprefix cacheが
効く順序を意図的に構成した。

@@RECALL_METHOD@@

\subsection{Serving最適化}
VLM推論を、動画・promptを内部表現へ変換するPrefillと、最終回答tokenを生成するDecodeへ
分けて計測した。動画入力ではPrefill負荷が大きくなりやすい一方、本手法は短いindex配列のみを
continuous batchingは、異なるrequestのPrefillとDecodeをGPU schedulerが逐次組み合わせ、
GPUの遊休時間を減らす機構である。本稿では同時投入上限をconcurrency 1/2/4/8と変化させ、
aggregate throughputとrequest単位のtail latencyを同時に測った。concurrencyを上げれば
GPU利用率は向上するが、KV cacheの競合、待ち行列、再計算によってP95 latencyやpreemptionが
悪化し得る。

Chunked Prefillは長いPrefillを一定token数のchunkへ分割し、他requestのDecodeを割り込ませる。
2,048および8,192 token budgetを比較し、短い応答を持つ24-frame workloadでも効果があるかを
検証した。また、同一video embeddingを再利用するmultimodal（MM）cacheと、共通prompt tokenを
再利用するprefix cacheを区別して記録した。cache効果を測るrunでは同一clipを連続投入する
\texttt{video-grouped-hot}、純粋なserving比較では順序を乱す\texttt{shuffled-cold}を用いた。

\subsection{公開自車運動contextとタグ限定fusion}
@@MOTION_METHOD@@

\subsection{Nano--Super cascade}
全件を高価なSuperへ投入せず、まずNanoで一括判定し、Nanoの予測だけから再確認対象を選ぶ
二段構成を評価した。routing条件は、positive数が3以上、歩行者系positiveを含む、
複雑時系列タグpositiveを含む、の三種である。routing判断にGTは使用せず、選択されたclipだけ
Superの出力へ置換した。この設計により、NanoのみとSuper全件の間に複数の実測運用点を構成する。

\section{実験設定}
developmentでは15 clipsのclass-balanced Smokeで明らかに劣る条件を除外し、その後
全169 clipsでsampling/promptを選定した。evaluation 875 clipsは条件固定後に一度だけ
Nano/Superへ適用した。temperature 0、seed 20260808とし、モデル、prompt、media、
container digest、instance、GPU telemetryおよび全SHA-256をlock artifactへ保存した。
Smokeは方式探索、developmentは方式選定、evaluationは最終性能推定という役割を持ち、
evaluation結果を見てsamplingやpromptを再調整しない三段階設計とした。さらに、
sampling比較ではpromptとmodelを固定し、serving比較ではmediaとpromptを固定することで、
複数要因を同時に変更した比較を避けた。

Recall原因診断は、developmentから低Recall 7タグについて、時間的にcoveredであった
False Negative 2件、True Positive 1件、negative control 1件をタグごとに選び、
重複を共有した24 clipsで実施した。この集合は難例を意図的に濃縮しているため、
診断上のP/R/F1をROAD母集団性能として扱わず、FN回復数、TP維持数、
negative-control FP数を主に報告する。
GT event時刻およびROAD boxを用いる条件は原因切り分け専用であり、Evaluationの最終方式へ
混入させていない。

\begin{center}
\captionof{table}{実験結果の証拠水準と許容する主張}
\label{tab:claim-levels}
\scriptsize
\begin{tabularx}{\columnwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
\toprule
水準 & 使用データ・GT利用 & 本稿で許容する主張\\
\midrule
Population evaluation & 未使用Evaluation 875 clips、推論入力へGT不使用 &
固定方式の母集団品質、分布誤差、費用\\
Development selection & 169 clips、方式・閾値の選定 &
候補間の選定根拠。母集団性能とはしない\\
Oracle diagnostic & 難例24 clips、時刻・box選択へGT使用 &
原因分解と到達可能性。実運用性能とはしない\\
Production proposal & detector/tracker等で候補を近似 &
実装仮説。固定Evaluationでの検証を今後要する\\
\bottomrule
\end{tabularx}
\end{center}

Nanoはg6e.2xlarge（1 GPU）、Superはg6.24xlarge（4 GPU）で実行した。後者は
GPU間がPCIe-onlyでvLLM custom all-reduceを利用できず、NCCL fallbackを含む実測である。
EC2定常推論費用は、artifactに記録したon-demand単価$p$とitem当たりwall時間$t$から
\[
 C_{1000}=p\frac{t}{3600}\times 1000
\]
で算出した。観測単価はus-west-2におけるg6e.2xlargeの
\$2.24208/hとg6.24xlargeの\$6.6752/hである。EKS control plane、EBS、NAT、
data transfer、model download、compile、idle時間は含めない。
したがって本稿の費用は、十分なrequestが継続して供給される定常状態のEC2下限に相当する。
低頻度運用ではmodel起動とidle時間を償却できず、実際の1 clip当たり費用は増加する。

\subsection{段階的実験設計と採用基準}
実験は、少数で方式の成立性を確かめるSmoke、未知条件へ進める方式を固定するdevelopment、
最終性能を一度だけ推定するevaluationに分けた。各段階の役割を混同しないため、
次の採用規則を設けた。
\begin{enumerate}
  \item Smokeではrequest failure、parse failure、preemptionおよび明らかな品質劣化を確認し、
        不安定な条件を全developmentへ拡張しない。
  \item Developmentではcoverage 99\%以上を必須とし、F1、MCC、Balanced Accuracyを主に、
        費用とlatencyを副に用いてPareto候補を選ぶ。複雑な方式は単純baselineを上回る場合だけ
        evaluation候補とする。
  \item Evaluationでは条件・閾値・promptを再調整しない。差はsource-video単位paired
        bootstrapで評価し、CIが0を跨ぐ場合は明確な改善と主張しない。
  \item 追加VLM passは、対象タグの改善が追加費用と誤変更を上回る場合だけ採用する。
        全体F1が僅かに上がっても、無関係タグの揺らぎが支配する条件は棄却する。
\end{enumerate}

\begin{center}
\captionof{table}{研究課題ごとの操作因子、固定条件および判断目的}
\label{tab:experiment-intent}
\scriptsize
\begin{tabularx}{\columnwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
\toprule
課題 & 変更した因子 & 固定した条件・判断目的\\
\midrule
RQ1 & Sampling、timestamp、prompt、request分割 &
同一clip・model・frame budget。品質差の原因を入力設計へ限定\\
RQ2 & 短い定義、全タグ詳細化、Hybrid Core &
同一169 clips・media・Super。詳細化範囲とPrompt長を比較\\
RQ3 & Nano / Super &
同一evaluation clips・media・prompt。model規模の効果を測定\\
RQ4 & concurrency、Prefill chunk、cache順序 &
同一15 clips・media・prompt。速度、tail、費用を分離\\
RQ5 & OPR/VO context、決定論的prior、Map pass &
映像baseline固定。改善・悪化pairと追加pass費用で採否\\
RQ6 & Oracle 24/48、ROI、track、tag score &
Development難例24 clips。時間・空間・競合を段階分解\\
RQ7 & 予測ラベルによる集計・ranking &
同一GT。分布誤差と人手確認濃縮率を分類品質と分離\\
\bottomrule
\end{tabularx}
\end{center}

\section{実験結果}
本節では、各実験を「検証意図、観測結果、解釈、次段階への採否」の順に記述する。
Smoke値は方式探索またはserving境界の確認に限って用い、最終品質の主張はFull Runに限定する。
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

表\ref{tab:ablation}の各行は、同一development clipsに対して入力frameの選び方、
timestamp表現、判定手続およびrequest分割だけを変更した条件である。
\texttt{uniform 24-reasoned}はPrecision 74.1、Recall 68.4、F1 71.1、MCC 0.621で
最良となった。これに対し、motion peakを用いる\texttt{hybrid}および
\texttt{adaptive}はRecallを同程度に維持したものの、PrecisionまたはMCCで上回らなかった。
画素変化の大きさは必ずしも意味的eventの重要度と一致せず、信号状態、停止、遠方の歩行者
といった低motion事象を均等samplingが安定して保持したためと考えられる。

\texttt{timestamp}のみの条件はPrecisionが高い一方、Recallが30--35\%程度へ低下した。
これはモデルが確実なpositiveだけを返す保守的な判定へ偏ったことを示す。
同じframe budgetでも\texttt{reasoned}手続を加えるとF1が約70\%へ上昇しており、
本問題ではframe数の増加より、全時刻と除外条件を照合させるprompt構成が支配的であった。
\texttt{grouped}はcache再利用を可能にするものの、タグ群ごとに文脈が分断され、
この条件では品質改善を示さなかった。以上から、Full Runには単一requestのreasoned条件を
採用し、uniformとadaptiveを最終候補とした。

@@PROMPT_RESULTS@@

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

表\ref{tab:quality}は、developmentで方式を固定した後に初めて実行した875 clipsの結果である。
全条件のcoverageが100\%であるため、差は出力欠落ではなく判定内容に起因する。
Nanoのuniform化はadaptive比でPrecisionを71.3から73.4へ改善した一方、Recallを66.7から
65.9へ低下させ、F1差は+0.5 pointsに留まった。paired bootstrapの95\% CIは
-0.2--+1.2 pointsで0を含むため、developmentで観測した優位が未知動画へ明確に一般化したとは
結論しない。ただし、単純なuniform方式が複雑なmotion解析と同等の品質を示したこと自体は、
前処理実装と運用保守を簡素化できる点で重要である。

SuperはNano adaptive比でPrecisionを10.6 points、Recallを1.6 points改善し、F1 74.5、
MCC 0.678に達した。改善の中心はfalse positiveの抑制であり、より大きなmodelが時系列eventを
一律に多く発見したというより、肯定条件と除外条件を厳密に区別した結果と解釈できる。
一方、定常EC2費用はNanoの約20倍であり、品質差だけから全件Superを選ぶことはできない。

\begin{center}
\centering
\captionof{table}{タグ別F1。``--''は陽性予測または有効分母がなくF1を定義できない場合を示す}
\label{tab:tags}
\small
\begin{tabular}{@@TAG_COLUMN_SPEC@@}
\toprule
タグ & GT陽性 & @@MODEL_HEADERS@@\\
\midrule
@@TAG_ROWS@@
\bottomrule
\end{tabular}
\end{center}

表\ref{tab:tags}は、micro平均では隠れるタグごとの難易度差を示す。自車走行・自車停止は
陽性例が多く、継続的な視覚変化も明瞭であるため、全modelでF1 88以上となった。
対象信号赤、自転車存在、二輪車存在ではSuperの誤認抑制が寄与した。一方、自車車線変更は
全条件でF1 15前後、歩行者横断待ちは30未満であり、静止画的な物体存在だけでなく、
lane境界との関係、対象trackの継続、将来意図を判定する必要がある。

Superは自車左折を28.2から59.8へ大きく改善したが、停止車両や自車右折ではNanoを下回った。
したがって、model規模の増加は全タグへ一様に作用しない。特に陽性数20の車線変更や34の
二輪車存在は推定分散が大きく、数pointsの順位を一般化すべきではない。運用上は、
全体F1だけでmodelを選ぶのではなく、収集目的となるタグの陽性数、タグ別F1および誤り費用を
併せて評価する必要がある。
\begin{multicols}{2}

@@RECALL_RESULTS@@

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

表\ref{tab:algorithm-runtime}の第2列は1 clipを判定するrequest数、第3列はrun全体の
経過時間をclip数で割った実効処理時間である。P95はrequest latencyの遅い側5\%点を表す。
PrefillとDecodeはvLLM metricsから得たstage時間、MM hitとPrefix hitはそれぞれ
動画特徴およびprompt prefixの再利用率である。実効処理時間は運用費用に直結する
throughput指標、P95は個々の利用者が経験し得る待ち時間、stage時間は最適化対象の所在を示す。

Uniform条件では1 FPSから8 FPSへ増やすにつれ、wall/clipと費用が概ね増加したが、
decode時間はほぼ一定であった。これは追加frameが主としてPrefill負荷を増やし、
短い回答形式によってDecode負荷がframe数から分離されていることを示す。
\texttt{adaptive 24-reasoned}は出力をpositive indexへ限定することで、同じ24-frame級でも
wall/clipを低く保った。\texttt{grouped}はMM cache 100\%、Prefix cache 93.1\%を達成したが、
3 requestのDecodeとscheduler overheadが累積し、単一reasoned requestより高価となった。
従って、cache hit率の高さだけを最適化目標にしてはならず、再利用後に残るrequest数と
Decode量を含むend-to-end時間で採否を決める必要がある。

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

表\ref{tab:serving-sweep}では、同一media、同一prompt、同一15 clipsを固定し、
serving設定だけを変更した。\texttt{c1}から\texttt{c8}は同時に処理待ちとするrequest数を表し、
値が大きいほどGPUへ仕事を供給しやすい一方、各requestの待ち時間とKV-cache消費が増える。
Chunked Prefillの2,048/8,192は、一度にschedulerへ渡すprefill token budgetであり、
modelの最大context長ではない。\texttt{preempt.}はKV-cache不足等により進行中sequenceを
退避・再計算した回数で、0であることを安定運用の重要なgateとした。

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

表\ref{tab:distribution}のprevalence MAEは、13タグそれぞれの出現率について、
予測とGTの絶対差を平均した値である。JSDはタグ構成全体の形状差を表し、0に近いほど
dataset分布を忠実に再現する。SuperはMAE 0.066、JSD 0.022でNanoを上回り、
個別clipの分類だけでなく、収集datasetの構成把握にも有利であった。NanoのMAE 0.084は、
大量一次走査には利用可能であるものの、数points単位の分布変化を監視する用途では
人手監査またはSuperによる補正が必要であることを示す。

@@CURATION_BLOCK@@

表\ref{tab:curation}は、予測positiveを先頭へ並べ、人が上位何\%を確認するかを変えた結果である。
\texttt{P@K}は確認対象の純度、\texttt{R@K}は全希少sceneのうち発見できた割合、
\texttt{enrichment}は同数を無作為抽出した場合に対する陽性密度の倍率を表す。
確認率を増やすほどR@Kは上昇する一方、確信度の低い候補まで含むためP@Kとenrichmentは低下する。
Superの上位1\%はP@K 88.9\%、enrichment 4.77であり、限られた人手予算を高濃度候補へ
集中できる。これはactive learningによる再学習効果そのものではなく、その前段となる
候補抽出効率を示す結果である。

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
@@MOTION_DISCUSSION@@
@@PROMPT_DISCUSSION@@
@@RECALL_DISCUSSION@@
開発splitではUniform 24-frameが最良であったが、evaluationでのF1改善は0.5 pointsに留まり、
信頼区間も0を跨いだ。従って「均等samplingが常に優れる」とは主張せず、motion peakへ
frameを集中するsamplingでは明確な品質改善が得られなかったという否定的結果を重視する。
一方、後述する構造化VO priorは有意な改善を示した。Uniformは前処理が
単純で費用予測も容易であるため、同等品質なら実装上の既定値として合理的である。

frame数を増やすだけでは性能は単調増加せず、timestampを提示した上で、全frameと除外条件を
順序立てて照合させるprompt形式の影響が大きかった。また、FNの95\%超はevent区間を
十分含む入力でも残った。従って、次段階の品質改善は一律なFPS増加ではなく、小物体crop、
detector/track summary、lane geometry、タグ別のcontrastive criteria、およびSuperによる
選択的再判定へ配分すべきである。

grouped requestはMM/prefix cacheを高率に再利用したが、request数とDecodeが増えるため、
end-to-end費用を削減しなかった。cacheは「複数回問い合わせる必要が既にある場合」の
追加費用を抑える機構であり、cache hitを得るためにrequestを分割すること自体は目的にならない。
@@TAG_FINDINGS@@

\subsection{品質を支配した要因の階層}
本実験で観測した改善幅は、すべて同じ種類の変更ではない。Promptはモデルが既に見ている
evidenceの照合方法、model規模は視覚・意味推論能力、VO priorは映像で曖昧な自車状態の
外部拘束、Samplingはどの時刻を観測するかに作用する。代表的な差を表
\ref{tab:effect-hierarchy}に整理する。

\begin{center}
\captionof{table}{代表的な変更の効果量と解釈}
\label{tab:effect-hierarchy}
\scriptsize
\begin{tabularx}{\columnwidth}{>{\raggedright\arraybackslash}Xrr>{\raggedright\arraybackslash}X}
\toprule
変更 & 基準F1 & 変更後F1 & 解釈\\
\midrule
timestamp-only→reasoned prompt（dev） & 45.8 & 70.2 &
既存evidenceの確認手続が最大の改善要因\\
Nano adaptive→Super（eval） & 68.9 & 74.5 &
主にfalse positive抑制。model能力差\\
Super→固定VO prior（eval） & 74.5 & 76.2 &
自車状態だけを構造化運動で補正\\
Nano adaptive→uniform（eval） & 68.9 & 69.4 &
小差でCIは0を跨ぐ。Sampling優位は未確定\\
\bottomrule
\end{tabularx}
\end{center}

この順序は、時系列scene labelingでは「より多くのframeを入れる」ことが第一の改善策ではなく、
まず判定手続と出力契約を安定させ、次にmodel能力、最後に残存誤りへ構造化contextを割り当てる
方が効率的であることを示す。ただし、効果量は異なるsplitと操作因子から得たものであり、
相互に加算可能な寄与率を表すものではない。

\subsection{生成的contextと決定論的fusion}
自車運動contextを自然言語としてSuperへ再入力するdevelopment実験では、映像baseline
F1 72.2に対し、VO contextによる自車action再判定はF1 74.6、27 pairs改善・8 pairs悪化であった。
一定の効果は確認できたが、動画Prefillを伴う追加requestが必要であり、contextが直接支持しない
タグも再生成させると予測が変動する。公開OPR軌跡は速度・道路曲率を粗く表す一方、
ROADの「交差点での右左折」と単なる道路湾曲を一意に区別できない。

これに対し固定VO priorは、変更対象を走行、停止、左折の3タグへ限定し、13タグ全体では
90 pairs改善・16 pairs悪化となった。追加VLM passを要さず、判断根拠と変更箇所を追跡できる。
この結果は、異種sensor情報をVLM promptへ無制限に足すより、各情報源の観測能力を定義し、
タグ単位のpromotion/vetoとして融合する方が安定することを示す。特に、粗いyawから右左折を
新規positiveへpromotionするより、視覚positiveを支持しない場合だけvetoする保守的利用が安全である。

\subsection{否定的結果から得られた設計知見}
本稿では採用しなかった条件も、方式選定上の重要な結果である。
\begin{itemize}
  \item GTを使わないmotion peak集中はUniformを明確に上回らなかった。一方、対象をcovered FNへ
        限定したSuperのGT Oracle 48は5/14件を回復した。従って局所時間密度は一部に有効だが、
        全件一律の高FPS化を正当化するほど支配的ではない。
  \item 48-frame化、multi-scale mosaic、GT-box ROIは回復と悪化が非単調であり、Prefill費用も
        増加した。48 framesはFull Run済みの最終方式ではなく、candidate routingを前提とする
        選択的extensionとして扱う必要がある。
  \item grouped requestは高いMM/prefix cache hitを得たが、3回分のDecodeとscheduler overheadにより
        単一requestより高価であった。cache hit率は最終目的ではない。
  \item Chunked Prefillは24-frame workloadでserver default c4を上回らず、c8は費用最小でも
        tail latencyとSuperのpreemptionを悪化させた。
  \item 車線変更専用passおよびMap付き4タグpassは追加費用に対して悪化pairが多く、Evaluationへ
        進めなかった。複雑なmaneuverにはprompt追加ではなくlane geometryとtrackの明示が必要である。
\end{itemize}
このように、複雑さを増す候補は単純baselineを上回った場合だけ次段階へ進めた。
結果として最終方式は、最も多くのcomponentを持つ構成ではなく、効果が再現したcomponentだけを
残した構成となった。

\subsection{推奨構成}
本実験範囲から、実証水準を分けて次の構成を提示する。
\begin{enumerate}
  \item \textbf{大量offline labeling:} Nano、Uniform 3 FPS/24 frames、
        reasoned single requestを用いる。throughputと費用を優先する場合はconcurrency 8、
        tail latencyを管理する場合はconcurrency 4とする。
  \item \textbf{Evaluationで実証済みの品質優先構成:} Super、Adaptive 24 frames、
        reasoned single request、concurrency 4を用いる。@@MOTION_RECOMMENDATION@@
        PCIe-only TP4であり、GPU推論費用はNanoの約20倍である。
  \item \textbf{品質・費用の中間:} 全件をNanoで処理し、複雑時系列positiveだけをSuperへ送る。
        本条件ではF1 73.4、\$10.01/1,000 clipsで、Super全件のF1 74.5へ近づいた。
  \item \textbf{次に固定Evaluationすべき提案extension:} 一次passへHybrid Coreを用い、
        detector/tracker等が強く支持するnegativeだけを48-frame・タグ別YES/NO passへ送る。
        これはDevelopmentおよびOracle診断に基づく提案であり、76.2\%へ上乗せした性能値は
        本稿では報告しない。
\end{enumerate}
Chunked Prefillは本workloadでserver defaultを上回らなかったため、既定設定を変更する根拠はない。
一方、長時間動画やframe budgetを増やす場合はPrefill比率が上昇するため、同じ結論を外挿せず
再測定する必要がある。

\subsection{運用費用の感度}
@@COST_PROJECTION@@

\subsection{再現性と実験妥当性}
品質評価は公開ROADだけから構成し、source video単位でdevelopment/evaluationを分離した。
evaluation条件はdevelopmentで固定後に一度だけ適用し、serving Smokeの品質値を
Full Run表へ混入させていない。全runは入力media、timestamp、prompt、container digest、
vLLM command、pod profile、GPU telemetryおよびSHA-256を保存した。表
\ref{tab:serving-sweep}は同一15 clipsに対する各1回の探索的計測であり、
latencyの信頼区間やinstance間変動を推定していない。この制約を越えて
小数点以下の差を一般化せず、採否は大きなthroughput差、tail latencyおよび
preemptionの有無に基づけた。さらにNano servingの主要6条件はpodを再作成して3回反復し、
平均と標本標準偏差を報告した。これにより、単一runの一時的なcache状態やscheduler揺らぎを
一般化する危険を低減した。

\subsection{限界}
ROADは英国Oxford周辺の限定されたcamera/domainであり、天候・地域・sensor placementの
一般化を保証しない。clip-level変換はevent onset/offsetのtemporal localization精度を
直接測らない。Hybrid Coreは3 source videosのdevelopmentで選定され、未使用evaluationへ
まだFull Runしていないため、75.6\%を母集団性能として扱えない。48-frame診断は同じ
developmentから意図的に難例を濃縮し、GT event時刻とGT boxを含む上限条件を使用した。
従って、診断P/R/F1、YES-score閾値、費用proxyは探索的結果であり、実運用のPrecision/Recallや
全clip費用を表さない。log probabilityから作ったYES scoreは校正済み確率ではない
\cite{guo2017calibration}。閾値も同じ診断集合上で比較したため、独立したcalibration splitで
再固定し、Brier scoreやECE等で校正性を確認する必要がある。
また、Hybrid候補3種と全Contrastiveを同一developmentで比較しており、多重比較を補正した
検定ではない。簡潔性を選択規則へ含めたものの、3 source videosに対するblock bootstrapの
cluster数は少なく、CI自体も粗い。従ってPrompt改善の母集団効果は固定Evaluationで確認すべきである。
希少タグはsource video数が少なくbootstrap CIが広い。Superの速度はPCIe-only 4 GPU
構成の制約を含み、NVLink/NVSwitch環境の上限性能ではない。これらを越える主張は行わない。

\section{結論}
本稿は、公開ROAD datasetの密な時間注釈から完全な正負GTを構成し、Cosmos 3による
時系列scene labelingを品質、計算費用、serving効率、分布推定および希少scene抽出の
観点から評価した。正例のみを信用する評価を避け、false positiveとabstentionを含む
confusion matrix、元動画単位CI、99\% coverage gateを導入した点が重要である。
結果は、入力frame数よりもtimestampを含む判定手順とprompt構成が品質へ大きく影響し、
同一mediaを用いる複数requestではcache再利用により追加費用を抑えられることを示した。
Developmentでは、全タグを一律に長文化するより、難しい5タグだけへ肯定証拠・時間検査・
除外条件を付与するHybrid Coreが、短いbaselineに対してF1を3.3 points改善した。
また、covered FNを対象とする上限診断ではGT event近傍48 framesにより5/14件、
タグ別の未校正YES scoreでは閾値に応じて7--11/14件を回復した一方、
negative-control FPも2--5/7件発生した。
この結果から、全件のframe数を増やすのではなく、外部候補が支持するnegativeだけへ
48-frame・tag-specific passを適用する階層方式を提案した。
また、ROAD上のcontrolled serving実測から、continuous batching、Chunked Prefill、
短いoutput、同一mediaのMM/prefix cache再利用が、互いに異なるstageへ作用することを
確認した。したがって、モデル選択だけでなく、evidence生成からrequest schedulingまでを
一体で設計する必要がある。
@@MOTION_CONCLUSION@@
今後は、Hybrid Coreと候補駆動48-frame passを未使用evaluationへ固定適用し、
detector/trackerがGT Oracle時刻をどこまで近似できるかを検証する。さらに、
異なる地域の公開dataset、event境界のtemporal mAP、独立calibration上の連続YES score、
および実際のdownstream再学習を含むactive learning loopで検証する。

{\footnotesize
\raggedright
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
\bibitem{oprproject2026}
OPR Project, ``OxfordRobotCar OpenPlaceRecognition,'' Hugging Face Datasets,
CC BY-NC-SA 4.0, accessed Aug. 9, 2026.
\bibitem{openstreetmap2026}
OpenStreetMap contributors, ``OpenStreetMap,'' Open Data Commons Open
Database License (ODbL), fixed snapshot accessed Aug. 9, 2026.
\bibitem{guo2017calibration}
C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger,
``On Calibration of Modern Neural Networks,''
in \textit{Proc. 34th Int. Conf. Machine Learning}, PMLR 70:1321--1330, 2017.
\end{thebibliography}
}
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
        "@@MOTION_ABSTRACT@@": str(motion_text["abstract"]),
        "@@MOTION_DISCUSSION@@": str(motion_text["discussion"]),
        "@@MOTION_CONCLUSION@@": str(motion_text["conclusion"]),
        "@@MOTION_RECOMMENDATION@@": str(
            motion_text["recommendation"]
        ),
        "@@MOTION_METHOD@@": motion_method,
        "@@PROMPT_METHOD@@": prompt_method,
        "@@PROMPT_RESULTS@@": prompt_results,
        "@@PROMPT_ABSTRACT@@": prompt_summary["abstract"],
        "@@PROMPT_DISCUSSION@@": prompt_summary["discussion"],
        "@@RECALL_METHOD@@": recall_method,
        "@@RECALL_RESULTS@@": recall_results,
        "@@RECALL_ABSTRACT@@": recall_summary["abstract"],
        "@@RECALL_DISCUSSION@@": recall_summary["discussion"],
        "@@TAG_FINDINGS@@": tag_findings,
        "@@COST_PROJECTION@@": cost_projection,
        "@@ANNOTATED_FRAMES@@": f"{int(audit['annotated_frame_count']):,}",
        "@@ANNOTATION_FRAMES@@": f"{int(audit['annotation_frame_count']):,}",
        "@@ANNOTATED_RATIO@@": _pct_text(
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
        "@@TAG_COLUMN_SPEC@@": "lr" + ("r" * len(evaluation_profiles)),
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


def _prompt_reasoning_method_tex(
    selection: dict[str, Any] | None,
    audit: dict[str, Any] | None,
) -> str:
    if not selection:
        return ""
    selected = selection["selected"]
    tags = [
        TAG_LABELS.get(str(tag), str(tag))
        for tag in selected.get("detailed_tags") or []
    ]
    tags_text = "、".join(_tex(tag) for tag in tags)
    baseline_words = _prompt_audit_words(audit, "road-reasoned")
    selected_words = int(selected.get("prompt_criteria_words") or 0)
    return rf"""\subsubsection*{{Hybrid Core Reasoning}}
短いbaseline定義はタグの意味を一文で与えるだけであった。これに対し詳細定義では、
各タグ$t$について、(i) 意味、(ii) 追跡すべき対象、(iii) positiveを支持する視覚証拠、
(iv) 複数timestamp間で確認すべき変化、(v) 類似事象を除外する反証、の五要素を
$D_t=(m_t,o_t,e_t,\Delta_t,r_t)$として明示した。モデルには自由記述の思考過程を
出力させず、全frameを時系列順に検査した後のpositive indexだけを返させる。
従って、本稿のReasoning強化は長い回答を生成するchain-of-thoughtではなく、
判定前に実施すべき証拠照合手順を入力契約として構造化する方法である。

全13タグを詳細化するとPrompt長とKV-cache消費が増え、厳しい除外条件が視覚的に単純な
タグのRecallを下げる可能性がある。そこで、Development上で全タグ詳細化と3種の部分詳細化を
同一media・同一Superで比較した。数値上の最高候補と簡潔な候補のpaired 95\% CIが0を跨ぐ
場合は短い方を選ぶparsimony規則を事前に採用し、最終的に
\texttt{{{_tex(str(selected['profile']))}}}を固定した。詳細化対象は{tags_text}の
{len(tags)}タグであり、残る{len(TAG_LABELS) - len(tags)}タグは短い定義を維持する。
criteria語数はbaselineの{baseline_words}語から{selected_words}語へ増えるが、
全タグ詳細化より短い。選定にはEvaluationを使用していない。"""


def _prompt_reasoning_results_tex(
    development: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    audit: dict[str, Any] | None,
) -> str:
    if not development or not selection:
        return ""
    selected_profile = str(selection["selected"]["profile"])
    preferred = [
        "road-adaptive-24-reasoned",
        "road-adaptive-24-contrastive-v2",
        "road-adaptive-24-hybrid-core-v1",
        "road-adaptive-24-hybrid-temporal-v1",
        "road-adaptive-24-hybrid-f1-v1",
    ]
    labels = {
        "road-adaptive-24-reasoned": "短いReasoned baseline",
        "road-adaptive-24-contrastive-v2": "全タグContrastive",
        "road-adaptive-24-hybrid-core-v1": "Hybrid Core",
        "road-adaptive-24-hybrid-temporal-v1": "Hybrid Temporal",
        "road-adaptive-24-hybrid-f1-v1": "Hybrid F1-selected",
    }
    profile_rows = {
        str(row["profile"]): row
        for row in development.get("profiles") or []
        if str(row.get("profile")) in preferred
    }
    rows = []
    for profile in preferred:
        row = profile_rows.get(profile)
        if row is None:
            continue
        variant = _profile_to_prompt_variant(profile)
        words = _prompt_audit_words(audit, variant)
        detailed = _prompt_audit_detailed_tags(audit, variant)
        micro = row["micro"]
        runtime = row["runtime"]
        label = labels[profile]
        if profile == selected_profile:
            label += "（採用）"
        rows.append(
            f"{_tex(label)} & {words} & {detailed} & "
            f"{_pct(micro['precision'])} & "
            f"{_pct(micro['strict_recall'])} & "
            f"{_pct(micro['strict_f1'])} & "
            f"{_number(micro['selective_mcc'])} & "
            f"{_number(runtime.get('prompt_tokens_mean'), 0)} & "
            f"{_number(runtime.get('request_e2e_seconds_p95'), 1)} \\\\"
        )
    selected = selection["selected"]
    ci = selected.get("paired_f1_delta_ci95") or []
    raw = selection.get("parsimony_comparison") or {}
    raw_ci = raw.get("raw_best_minus_reference_f1_ci95") or []
    selected_delta = (
        float(selected["metrics"]["strict_f1"])
        - float(
            profile_rows["road-adaptive-24-reasoned"]["micro"][
                "strict_f1"
            ]
        )
    )
    ci_text = (
        f"{100 * float(ci[0]):+.2f}--{100 * float(ci[1]):+.2f}"
        if len(ci) == 2
        else "算出不能"
    )
    raw_ci_text = (
        f"{100 * float(raw_ci[0]):+.2f}--{100 * float(raw_ci[1]):+.2f}"
        if len(raw_ci) == 2
        else "算出不能"
    )
    return (
        r"""\subsection{タグ選択型Reasoning Prompt}
表\ref{tab:prompt-selection}は、同一Development 169 clips、同一24-frame media、
同一Cosmos 3 Super、temperature 0で、詳細定義を適用するタグ集合だけを変更した結果である。
候補3条件は同一run内でランダムに混在させ、特定Promptだけがwarm stateを先取りする偏りを
抑えた。P95はこの混在runにおける参考値であり、独立cold runの費用比較ではない。
\end{multicols}
\begin{center}
\captionof{table}{DevelopmentにおけるReasoning Promptの選定}
\label{tab:prompt-selection}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Prompt & criteria語数 & 詳細タグ & P & R & F1 & MCC & tokens & P95[s]\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{center}
\begin{multicols}{2}
短いbaselineに対し、全タグContrastiveはRecallを改善したが、Promptを一律に長文化した。
Hybrid CoreはF1を"""
        + f"{100 * selected_delta:+.2f}"
        + r""" points改善し、source-video単位paired bootstrap 95\% CIは"""
        + ci_text
        + r""" pointsで0を跨がなかった。数値上はHybrid Temporalが0.21 points高かったが、
Coreとの差の95\% CIは"""
        + raw_ci_text
        + r""" pointsであり、統計的に区別できなかった。このため、criteriaが22.5\%短く、
詳細化対象も少ないHybrid Coreを採用した。この選択は「最大値を出したPrompt」を
後追いで採るのではなく、同等群から複雑さの小さい方式を選ぶことでDevelopmentへの
過適合を抑える判断である。なお、この75.6\%はDevelopment値であり、Evaluation性能ではない。"""
    )


def _prompt_reasoning_summary_text(
    development: dict[str, Any] | None,
    selection: dict[str, Any] | None,
) -> dict[str, str]:
    if not development or not selection:
        return {"abstract": "", "discussion": ""}
    selected = selection["selected"]
    profiles = {
        str(row["profile"]): row
        for row in development.get("profiles") or []
    }
    baseline = profiles.get("road-adaptive-24-reasoned")
    if baseline is None:
        return {"abstract": "", "discussion": ""}
    baseline_f1 = float(baseline["micro"]["strict_f1"])
    selected_f1 = float(selected["metrics"]["strict_f1"])
    ci = selected.get("paired_f1_delta_ci95") or []
    ci_text = (
        f"{100 * float(ci[0]):+.1f}--{100 * float(ci[1]):+.1f} points"
        if len(ci) == 2
        else "算出不能"
    )
    return {
        "abstract": (
            "Developmentでは、難しい5タグだけを詳細化するHybrid Core Reasoningが、"
            f"短いPromptのF1 {_pct(baseline_f1)}を"
            f"{_pct(selected_f1)}へ改善した。"
        ),
        "discussion": (
            "Hybrid CoreのDevelopmentにおけるbaseline比F1差は"
            f"{100 * (selected_f1 - baseline_f1):+.1f} points"
            f"（95\\% CI: {ci_text}）であった。"
            "改善は、全タグを長文化することではなく、左折、車線変更、停止車両、"
            "横断待ち、二輪車の判定に必要な時系列証拠と反証を明示した結果である。"
            "ただし未使用Evaluationへはまだ適用していないため、Population性能の"
            "上昇としては主張しない。"
        ),
    }


def _prompt_audit_words(
    audit: dict[str, Any] | None,
    variant: str,
) -> int:
    row = _prompt_audit_variant(audit, variant)
    return int(row.get("criteria_words") or 0) if row else 0


def _prompt_audit_detailed_tags(
    audit: dict[str, Any] | None,
    variant: str,
) -> int:
    row = _prompt_audit_variant(audit, variant)
    if not row:
        return 0
    return sum(
        1
        for tag in row.get("tags") or []
        if int(tag.get("evidence_items") or 0) > 0
    )


def _prompt_audit_variant(
    audit: dict[str, Any] | None,
    variant: str,
) -> dict[str, Any] | None:
    if not audit:
        return None
    return next(
        (
            row
            for row in audit.get("variants") or []
            if str(row.get("variant")) == variant
        ),
        None,
    )


def _profile_to_prompt_variant(profile: str) -> str:
    return {
        "road-adaptive-24-reasoned": "road-reasoned",
        "road-adaptive-24-contrastive-v2": (
            "road-reasoned-contrastive-v2"
        ),
        "road-adaptive-24-hybrid-core-v1": (
            "road-reasoned-hybrid-core-v1"
        ),
        "road-adaptive-24-hybrid-temporal-v1": (
            "road-reasoned-hybrid-temporal-v1"
        ),
        "road-adaptive-24-hybrid-f1-v1": (
            "road-reasoned-hybrid-f1-v1"
        ),
    }.get(profile, profile)


def _selective_recall_method_tex(
    diagnostic: dict[str, Any] | None,
) -> str:
    if not diagnostic:
        return ""
    target_tags = [
        TAG_LABELS.get(str(tag), str(tag))
        for tag in diagnostic.get("selection", {}).get("target_tags") or []
    ]
    target_text = "、".join(_tex(tag) for tag in target_tags)
    return (
        r"""\subsection{候補駆動48-frame Recall refinement}
一次passは13タグを一括して高Precisionに判定する一方、negative出力には、証拠不足、
タグ間競合、実効判定閾値の高さが混在する。そこで本稿は、全clipを一律に高FPS化せず、
軽量detector/tracker、lane、VOなどの外部証拠がタグ$t$を支持し、一次予測がnegativeである
場合だけrouting変数$q_t(v)=1$とする条件付きsecond passを提案する。
\[
 q_t(v)=\mathbb{1}\!\left[\hat y_t(v)=0\ \land\
 c_t(v)\ge\gamma_t\right],
\]
ここで$c_t(v)$はタグ固有candidate score、$\gamma_t$はDevelopmentで固定する閾値である。
$q_t(v)=1$のとき、candidate時刻の前後を8 FPSで密にし、全域contextを残した48 frames
$M_{48,t}(v)$を生成する。48枚は追加し続けるのではなく、低優先の全域frameと置換して
budgetを固定する。

second passは対象タグだけをYES/NOで問い合わせ、先頭tokenのlog probabilityから
\[
 s_t(v)=\frac{\exp \ell_{\mathrm{YES}}}
 {\exp \ell_{\mathrm{YES}}+\exp \ell_{\mathrm{NO}}}
\]
を得る。$s_t$は校正済み確率ではなく、YES/NO間の相対的な未校正scoreである。
最終判定は独立calibrationで固定するタグ別閾値$\eta_t$と、lane境界横断、相対速度、person--road交差、
自車laneに対応する信号、brake-lamp変化等の構造化veto/promotionで決める。
13タグすべてを個別callする設計ではなく、candidateになったタグだけを再確認する。

この本番形の候補生成器はまだFull Runしていないため、本稿では"""
        + target_text
        + "の"
        + str(len(target_tags))
        + r"""タグを対象とし、GT event時刻、GT box、匿名trackを順に用いる
Oracle診断で各情報源の上限を分解した。GT action/class名はPromptへ渡していない。
Oracle 48の回復はcandidate-guided 48-frame設計の可能性を示すが、本番性能の証明ではない。"""
    )


def _selective_recall_results_tex(
    diagnostic: dict[str, Any] | None,
    baseline_confidence: dict[str, Any] | None,
    oracle_confidence: dict[str, Any] | None,
) -> str:
    if not diagnostic:
        return ""
    profile_labels = {
        "road-recall-baseline-24-core": "Baseline 24",
        "road-recall-oracle-24-core": "Oracle時刻 24",
        "road-recall-oracle-48-core": "Oracle時刻 48",
        "road-recall-oracle-48-multiscale-core": "48＋multiscale",
        "road-recall-oracle-48-roi-core": "48＋GT-box ROI",
        "road-recall-oracle-48-roi-motion-core": "ROI＋VO",
        "road-recall-oracle-48-roi-track-motion-core": (
            "ROI＋匿名track＋VO"
        ),
    }
    hard_rows = []
    for row in diagnostic.get("profiles") or []:
        hard_rows.append(
            f"{_tex(profile_labels.get(str(row['profile']), str(row['profile'])))} & "
            f"{int(row['original_false_negatives_recovered'])}/"
            f"{int(row['original_false_negative_count'])} & "
            f"{int(row['original_true_positives_retained'])}/"
            f"{int(row['original_true_positive_count'])} & "
            f"{int(row['negative_control_false_positives'])}/"
            f"{int(row['negative_control_count'])} & "
            f"{_number(row['mean_selected_frames'], 1)} & "
            f"{_number(row['mean_e2e_seconds'], 2)} & "
            f"{_number(row['estimated_usd_per_1000_proxy'], 2)} \\\\"
        )
    confidence_rows = []
    for media, summary in (
        ("Baseline 24", baseline_confidence),
        ("Oracle 48", oracle_confidence),
    ):
        if not summary:
            continue
        thresholds = {
            round(float(row["threshold"]), 2): row
            for row in summary.get("confidence", {}).get("thresholds") or []
        }
        for threshold in (0.05, 0.20, 0.40):
            row = thresholds.get(threshold)
            if row is None:
                continue
            confidence_rows.append(
                f"{_tex(media)} & {threshold:.2f} & "
                f"{int(row['original_false_negatives_recovered'])}/14 & "
                f"{int(row['original_true_positives_retained'])}/7 & "
                f"{int(row['negative_control_false_positives'])}/7 & "
                f"{_pct(row['precision'])} & {_pct(row['recall'])} \\\\"
            )
    hard_table = "\n".join(hard_rows)
    confidence_table = "\n".join(confidence_rows)
    return (
        r"""\subsection{48-frame・ROI・未校正YES scoreによるRecall上限診断}
本診断はDevelopmentから意図的に抽出した24 clips、28 clip--tag pairsであり、
ROAD母集団の性能推定ではない。各低Recallタグについて、時間的にcoveredなFN 2件、
TP 1件、negative control 1件を選び、hard-label条件では14 FNの回復、7 TPの維持、
7 controlsに対するFPを数えた。
\end{multicols}
\begin{center}
\captionof{table}{低Recallタグに対する入力証拠の上限診断}
\label{tab:recall-diagnostic}
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
条件 & FN回復 & TP維持 & control FP & frames & E2E[s] & USD/1k proxy\\
\midrule
"""
        + hard_table
        + r"""
\bottomrule
\end{tabular}
\end{center}
\begin{multicols}{2}
同じ24-frame budgetでGT event周辺へ再配置すると2/14件、48 framesへ増やすと5/14件を
回復し、TP 7/7を維持した。従って、event frameを1枚以上含むことと、行動を判定できる
局所時間密度を持つことは同義ではない。一方、GT時刻を知る48 framesでも9件はhard labelの
まま残り、時間密度だけで見逃しを解消する仮説は棄却された。

GT-box ROIやmosaicは一部を回復したがTPも失った。cropは元画素を増やさず、mosaic化は
全体contextを変形するためである。匿名track＋VOは停止車両に有効だったが、自然言語contextを
全タグへ混ぜると無関係タグが揺れた。従って、本番ではfull-frameを保持し、candidate trackの
高解像度still/sequenceを追加し、構造化証拠は対象タグだけへ融合すべきである。

\end{multicols}
\begin{center}
\captionof{table}{タグ別の未校正YES/NO scoreによる探索的閾値比較}
\label{tab:recall-confidence}
\scriptsize
\begin{tabular}{lrrrrrr}
\toprule
Media & threshold & FN回復 & TP維持 & control FP & 診断P & 診断R\\
\midrule
"""
        + confidence_table
        + r"""
\bottomrule
\end{tabular}
\end{center}
\begin{multicols}{2}
Oracle 48の閾値0.05では11/14 FNを回復したが、control FPも5/7となった。閾値0.20では
8/14回復、3/7 FPとなり、RecallとPrecisionの交換を明示的に制御できた。
Baseline 24でもタグ別判定だけで多くのFNが回復したことから、見逃しには視覚証拠不足だけでなく、
13タグ同時判定におけるtag competition、positive index抑制、実効閾値が関与する。
ただし$s_t$はlog probabilityをYES/NO間で正規化した未校正scoreである。閾値は同じ
小規模診断集合で比較した探索値であり、独立calibrationなしに本番へ固定してはならない。
費用列はmean request latencyをconcurrencyで割ったqueue-adjusted
proxyで、1,000 clipsではなく1,000 routed tag checks相当の参考値である。"""
        + r"""

\begin{center}
\captionof{table}{低Recallタグの診断結果と本番で必要となる追加証拠}
\label{tab:recall-tag-diagnosis}
\scriptsize
\begin{tabularx}{\columnwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}
\toprule
タグ & 診断で観測した制約 & 本番で近似すべき証拠\\
\midrule
車線変更 & 時刻を増やしてもlane移行を同定できない &
lane境界横断、道路方向に対する横軌跡\\
横断待ち & 人物の存在だけでは意図を決められない &
person track、縁石距離、身体方向、停止時間\\
歩行者横断 & 小物体とego motionで移動方向が曖昧 &
person--road ROI交差、進入・退出時系列\\
停止車両 & 匿名track追加で2/2 FNを回復 &
ego補償済み相対速度、stationary duration\\
青信号 & crop後も元画素と自車lane対応が不足 &
高解像度信号crop、lane--signal association\\
車両制動 & 尾灯、反射、減速度を区別できない &
rear cropの点灯差分と相対減速度\\
自転車 & 遠方で二輪車・人物との識別が不安定 &
class-aware detector cropと同一trackの複数時刻\\
\bottomrule
\end{tabularx}
\end{center}"""
    )


def _selective_recall_summary_text(
    diagnostic: dict[str, Any] | None,
    oracle_confidence: dict[str, Any] | None,
) -> dict[str, str]:
    if not diagnostic:
        return {"abstract": "", "discussion": ""}
    oracle = next(
        (
            row
            for row in diagnostic.get("profiles") or []
            if row.get("profile") == "road-recall-oracle-48-core"
        ),
        None,
    )
    if oracle is None:
        return {"abstract": "", "discussion": ""}
    abstract = (
        "Development難例診断では、GT event近傍を48 framesへ高密度化して"
        f"{int(oracle['original_false_negatives_recovered'])}/"
        f"{int(oracle['original_false_negative_count'])} FNを回復した。"
    )
    if oracle_confidence:
        abstract += "タグ別の未校正YES scoreはRecall--Precision交換を制御できた。"
    return {
        "abstract": abstract,
        "discussion": (
            "Recall診断は、Full Runでcoveredと分類されたFNにも、局所時間密度不足、"
            "小物体・track不足、multi-label競合という複数原因が残ることを示した。"
            "Oracle 48のhard labelは5/14回復に留まる一方、タグ別scoreではより多くを"
            "回復できたため、一律なFPS増加よりcandidate routingとtag-specific判定を"
            "組み合わせる方が妥当である。ただし本診断は難例濃縮かつGT Oracleであり、"
            "本番候補生成器のRecallと全体費用は未測定である。"
        ),
    }


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
    motion_prior: dict[str, Any] | None,
    composed_motion: dict[str, Any] | None,
    context_fusion: dict[str, Any] | None,
    opr_motion: dict[str, Any] | None,
    visual_motion: dict[str, Any] | None,
    map_context: dict[str, Any] | None,
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
\end{center}
\texttt{any}はGT event区間内のframeを1枚以上含む率、
\texttt{2 frames}は時系列変化を比較できる最低2枚を含む率、
\texttt{span}は選択frameがevent内で0.5秒以上に広がる率、
\texttt{context}はevent開始前と終了後の双方を含む率である。
Hybridは2-frame coverageとspanで最良、Adaptiveはboundary contextで最良となった。
Uniformはcontextが89.2\%へ低下した一方、development F1では最良であった。
これは、event境界を機械的に多く含めることと、VLMが正しく意味判定することが同値では
ないことを示す。motion peakへの過度な集中より、clip全域の均等な観測が有効であった。"""
        )
    if error_analysis:
        rows = "\n".join(
            (
                f"{_tex(str(row['model']).split('/')[-1] + ' / ' + _short_profile(str(row['profile'])))} & "
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
断定しない。

Nano adaptiveでは904 FN中866件、Nano uniformでは926 FN中887件、Superでは861 FN中
819件がcovered missであった。すなわち、単純なframe追加によって直接回復し得る誤りは
全体の約5\%未満である。実画像の監査では、遠方の信号・歩行者、自転車の小さな投影面積、
夜間・逆光、および停止と減速のtaxonomy境界が含まれた。従って次の改善対象は、
一律なFPS増加ではなく、候補物体crop、track summary、道路幾何、タグ別prompt、
またはmodel規模である。"""
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
\noindent\begin{minipage}{\textwidth}
\refstepcounter{table}\label{tab:serving-repeat}
\begin{center}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrr}
\multicolumn{7}{c}{\textbf{\tablename~\thetable: ROAD 15 clipsを用いたNano serving 3反復
（平均$\pm$標本標準偏差）}}\\[3pt]
\toprule
条件 & runs & wall/clip[s] & P95[s] & prefill[s] & decode[s] & USD/1k\\
\midrule
"""
            + "\n".join(rows)
            + r"""
\bottomrule
\end{tabular}}
\end{center}
\end{minipage}
\begin{multicols}{2}
3反復でもdefault c4はwall/clip $1.38\pm0.01$秒、費用
\$0.862$\pm0.003$/1,000 clipsと安定した。c8は$1.22\pm0.04$秒まで短縮したが、
P95は$13.78\pm1.19$秒へ増加した。Chunked Prefillはdefault c4を上回らなかった。
従ってc4を対話的運用、c8をoffline batchの費用最小点とする。"""
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
全件SuperとNanoのみの間に、実測された品質・費用の中間運用点を構成できる。

歩行者系positiveだけを再判定する方策は全clipの23.7\%をSuperへ送り、F1を69.4から
70.3へ改善し、費用は\$3.88/1,000 clipsとなった。複雑時系列positiveをroutingする方策は
69.1\%をSuperへ送り、F1 73.4を\$10.01で達成した。これは全件SuperのF1 74.5、
\$13.48に近い品質を25.7\%低い費用で得る中間点である。一方、Nanoの見逃しはrouting条件を
発火しないため、cascadeはNanoのfalse negativeを完全には救済できない。運用方策は
誤検出削減を重視するか、見逃し削減を重視するかに応じて選ぶ必要がある。"""
        )
    if motion_prior:
        blocks.append(
            _public_motion_results_tex(
                motion_prior,
                composed_motion,
                context_fusion,
                opr_motion,
                visual_motion,
                map_context,
            )
        )
    return "\n\n".join(blocks)


def _public_motion_method_tex(
    motion_prior: dict[str, Any] | None,
    opr_motion: dict[str, Any] | None,
    visual_motion: dict[str, Any] | None,
) -> str:
    if not motion_prior:
        return (
            "本稿では追加の自車運動fusionを用いず、映像予測だけを評価する。"
        )

    stride = int((visual_motion or {}).get("stride_frames") or 5)
    width = int((visual_motion or {}).get("resize_width") or 480)
    vo_clips = int((visual_motion or {}).get("clip_count") or 0)
    vo_videos = int((visual_motion or {}).get("source_video_count") or 0)
    opr_clips = int((opr_motion or {}).get("clip_count") or 0)
    opr_videos = int((opr_motion or {}).get("source_video_count") or 0)

    changed_rules = []
    for row in motion_prior.get("tags") or []:
        rule = row.get("rule") or {"kind": "identity"}
        if rule.get("kind") != "identity":
            changed_rules.append(
                (
                    TAG_LABELS.get(str(row["tag"]), str(row["tag"])),
                    _motion_rule_description(rule),
                )
            )
    rule_rows = "\n".join(
        f"{_tex(tag)} & {_tex(description)} \\\\"
        for tag, description in changed_rules
    )
    if not rule_rows:
        rule_rows = "対象タグ & Developmentで改善した規則だけを採用 \\\\"

    coverage_text = (
        f"公開OPR派生軌跡はquality gate後に{opr_clips} clips・"
        f"{opr_videos} source videosをカバーした。"
        if opr_clips
        else "公開OPR派生軌跡は取得可能な走行だけを補助的に用いた。"
    )
    vo_text = (
        f"video-derived VOは{vo_clips} clips・{vo_videos} source videosへ"
        "同一条件で適用した。"
        if vo_clips
        else "video-derived VOは全対象動画へ同一条件で適用した。"
    )
    changed_count = len(changed_rules)
    total_tags = len(motion_prior.get("tags") or [])
    unchanged_count = max(total_tags - changed_count, 0)

    return (
        r"""公式Oxford RobotCarの生INS/GNSSは学術機関認証を要するため使用せず、
認証不要のOPR公開派生物\cite{oprproject2026}とROAD動画だけから再計算可能なVOを用いた。
OPR画像とROAD frameはperceptual hashで対応付け、複数anchor間のframe時刻を区分線形補間した。
frame coverage 70\%以上かつleave-one-anchor-out時刻誤差P95 2秒以下を同期quality gateとし、
条件を満たさない走行は外挿せずVOへfallbackした。"""
        + coverage_text
        + vo_text
        + rf"""

VOでは元動画を{stride} frames間隔、幅{width} pxへ縮小して走査した。空や道路中央の
移動物体に偏らないよう背景寄りの領域から最大800個のShi--Tomasi特徴点を抽出し、
Lucas--Kanade法で前後追跡した。forward--backward誤差1.5 px以下かつ20 tracks以上を
有効pairとし、median flow、0.8 px未満のlow-motion pair率、Essential Matrixの
RANSAC inlierおよび累積visual yawをclip単位へ要約した。yawはinlier率0.25以上のpairだけを
集計し、全pairの50\%以上が有効なclipだけをfusion対象とした。

生成的fusionでは、これらの数値を自然言語contextとしてSuperへ再入力し、自車actionを
再判定させた。一方、提案する決定論的fusionでは、映像予測を保持したまま、development上の
grid searchでF1、MCC、BAが改善し、かつ改善pair数が悪化pair数以上となる規則だけを採用した。
条件を満たさないタグはidentityへ戻した。Evaluationでは閾値を変更せず、追加VLM requestを
発生させない。採用規則は次の通りであり、{changed_count}タグだけを変更し、残る
{unchanged_count}タグは映像出力を保持した。

\begin{{center}}
\captionof{{table}}{{Developmentで固定したVO motion prior規則}}
\label{{tab:motion-rules}}
\scriptsize
\begin{{tabularx}}{{\columnwidth}}{{l>{{\raggedright\arraybackslash}}X}}
\toprule
対象タグ & 映像予測を変更する条件\\
\midrule
"""
        + rule_rows
        + r"""
\bottomrule
\end{tabularx}
\end{center}

このタグ限定設計は、motion情報が歩行者、物体、信号色などへ直接の根拠を持たないという
帰納biasを明示する。すなわち、VOをVLMの代替分類器として用いるのではなく、映像だけでは
曖昧な自車状態に対する支持またはvetoとして用いる。"""
    )


def _motion_rule_description(rule: dict[str, Any]) -> str:
    kind = str(rule.get("kind"))
    if kind == "moving_state":
        return (
            "no→yes: median flowが"
            f"{_number(rule['promote_flow_at_least'], 2)} px以上かつ "
            "low-motion率が"
            f"{_number(rule['promote_low_motion_at_most'], 2)}以下。"
            "yes→no: flowが"
            f"{_number(rule['veto_flow_at_most'], 2)} px以下かつ "
            "low-motion率が"
            f"{_number(rule['veto_low_motion_at_least'], 2)}以上"
        )
    if kind == "stopped_state":
        return (
            "no→yes: flowが"
            f"{_number(rule['promote_flow_at_most'], 2)} px以下、または "
            "low-motion率が"
            f"{_number(rule['promote_low_motion_at_least'], 2)}以上。"
            "yes→no: flowが"
            f"{_number(rule['veto_flow_at_least'], 2)} px以上かつ "
            "low-motion率が"
            f"{_number(rule['veto_low_motion_at_most'], 2)}以下"
        )
    if kind == "turn_veto":
        direction = "正" if rule.get("direction") == "left" else "負"
        return (
            "yesを維持するのは累積visual yawが"
            f"{direction}方向へ{_number(rule['minimum_signed_yaw_degrees'], 1)}°"
            "以上の場合だけ"
        )
    return "映像予測を変更しない"


def _public_motion_results_tex(
    motion_prior: dict[str, Any],
    composed_motion: dict[str, Any] | None,
    context_fusion: dict[str, Any] | None,
    opr_motion: dict[str, Any] | None,
    visual_motion: dict[str, Any] | None,
    map_context: dict[str, Any] | None,
) -> str:
    baseline = motion_prior["baseline"]["metrics"]
    prior = motion_prior["fused"]["metrics"]
    final_name = "Motion prior"
    final = prior
    final_delta = motion_prior.get("paired_f1_delta") or {}
    composed_is_evaluation = bool(
        composed_motion
        and composed_motion.get("split") == motion_prior.get("split")
    )
    if composed_is_evaluation:
        assert composed_motion is not None
        final_name = "Motion prior + scoped VO pass"
        final = composed_motion["composed"]["metrics"]
        final_delta = composed_motion["composed"].get(
            "paired_f1_delta_vs_baseline"
        ) or {}

    rows = [
        (
            "Super映像baseline",
            baseline,
            1,
            {"improved_pairs": 0, "regressed_pairs": 0},
        ),
        (
            "固定VO motion prior",
            prior,
            1,
            motion_prior["paired_changes"],
        ),
    ]
    if composed_is_evaluation:
        assert composed_motion is not None
        rows.append(
            (
                "Motion prior + 車線変更pass",
                final,
                2,
                composed_motion["composed"][
                    "paired_changes_vs_baseline"
                ],
            )
        )
    table_rows = "\n".join(
        (
            f"{_tex(name)} & {passes} & "
            f"{_pct(metrics['precision'])} & "
            f"{_pct(metrics['strict_recall'])} & "
            f"{_pct(metrics['strict_f1'])} & "
            f"{_number(metrics['selective_mcc'])} & "
            f"{int(changes['improved_pairs'])}/"
            f"{int(changes['regressed_pairs'])} \\\\"
        )
        for name, metrics, passes, changes in rows
    )
    opr_text = (
        f"公開OPR軌跡は{int(opr_motion['clip_count'])} clips、"
        f"{int(opr_motion['source_video_count'])} source videosを"
        "quality gate後にカバーした。"
        if opr_motion
        else "公開OPR軌跡は利用可能な走行だけに適用した。"
    )
    vo_text = (
        f"video-derived VOは{int(visual_motion['clip_count'])} clips、"
        f"{int(visual_motion['source_video_count'])} source videosをカバーした。"
        if visual_motion
        else "video-derived VOは全対象動画へ同一条件で適用した。"
    )
    context_text = ""
    if context_fusion:
        vo_rows = [
            row
            for row in context_fusion["rows"]
            if row.get("context_profile")
            == "road-adaptive-24-reasoned-visual-motion-ego"
            and row.get("policy") == "full_context"
        ]
        if vo_rows:
            row = vo_rows[0]
            context_text = (
                "developmentの自車action 5タグ再判定はF1 "
                f"{_pct(row['metrics']['strict_f1'])}、"
                f"改善/悪化{int(row['paired_correctness']['improved_pairs'])}/"
                f"{int(row['paired_correctness']['regressed_pairs'])} pairsであった。"
                "無関係タグへの揺らぎを避けるため、全文置換は採用しなかった。"
            )
    if composed_motion and not composed_is_evaluation:
        development_prior = (
            composed_motion.get("motion_prior", {}).get("metrics")
            or prior
        )
        scoped = composed_motion["composed"]["metrics"]
        scoped_delta = composed_motion["composed"][
            "paired_changes_vs_motion_prior"
        ]
        scoped_cost = composed_motion.get("context_runtime") or {}
        if _is_map_context_pass(composed_motion):
            snapshot = (map_context or {}).get("osm_snapshot") or {}
            snapshot_time = snapshot.get(
                "timestamp_osm_base", "固定時点"
            )
            context_text += (
                "さらに、公開OPR軌跡を固定OpenStreetMap snapshot"
                "\\cite{openstreetmap2026}"
                f"（{_tex(snapshot_time)}、ODbL）へmap matchingし、"
                "交差点距離、分岐数、信号設備距離、軌跡方位変化を4タグ専用passへ"
                "入力した。信号色は映像だけで判定した。同一development subsetで"
                f"F1 {_pct(development_prior['strict_f1'])}から"
                f"{_pct(scoped['strict_f1'])}へ低下し、"
                f"{int(scoped_delta['improved_pairs'])} pairs改善に対して"
                f"{int(scoped_delta['regressed_pairs'])} pairs悪化した。追加費用は"
                f"\\${_number(scoped_cost.get('estimated_usd_per_1000_clips'), 2)}"
                "/1,000 target clipsであったため、Full Run候補から除外した。"
            )
        else:
            context_text += (
                "さらに、車線変更1タグだけへ質問を集中したdevelopment passは、"
                "同じdevelopment splitにおけるmotion prior単独F1 "
                f"{_pct(development_prior['strict_f1'])}に対して"
                f"{_pct(scoped['strict_f1'])}へ悪化し、"
                f"{int(scoped_delta['improved_pairs'])} pairs改善に対して"
                f"{int(scoped_delta['regressed_pairs'])} pairs悪化した。"
                "追加費用は"
                f"\\${_number(scoped_cost.get('estimated_usd_per_1000_clips'), 2)}"
                "/1,000 clipsであり、評価Full Run候補から除外した。"
            )
    ci = final_delta.get("ci95")
    delta_text = (
        f"baselineに対するF1差の95\\% CIは"
        f"{100 * float(ci[0]):+.1f}--{100 * float(ci[1]):+.1f} pointsである。"
        if ci
        else ""
    )
    return (
        r"""\subsection{公開motion contextと決定論的fusion}
公式Oxford RobotCarの生INS/GNSSは学術機関認証を要するため、本稿では使用しない。
代わりに、認証不要のOPR公開派生物\cite{oprproject2026}に含まれるtimestamp画像と
粗いRTK軌跡を用いた。
ROAD frameと公開画像をperceptual hashで対応付け、anchor間を区分線形補間した。
frame coverage 70\%以上かつleave-one-anchor-out時刻誤差P95 2秒以下を同期quality gateとし、
条件を満たさない走行は外挿せずvideo-derived Visual Odometry（VO）へfallbackした。"""
        + opr_text
        + vo_text
        + r"""

VOは静的背景領域のLucas--Kanade feature track、forward--backward整合性、
Essential MatrixのRANSAC inlierから、median flow、low-motion率および累積visual yawを
clip単位へ要約する。これらをpromptへ自然言語として加える条件に加え、
developmentだけで閾値を固定し、映像baselineの自車走行・停止・左折だけを補正する
motion priorを評価した。evaluationでは閾値を再調整せず、追加VLM requestも発生しない。
"""
        + context_text
        + r"""
\begin{center}
\captionof{table}{Evaluationにおける公開VO motion fusion}
\label{tab:public-motion}
\scriptsize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
方式 & pass & P & R & F1 & MCC & 改善/悪化\\
\midrule
"""
        + table_rows
        + r"""
\bottomrule
\end{tabular}}
\end{center}
固定motion priorはSuper baselineのF1 """
        + _pct(baseline["strict_f1"])
        + "から"
        + _pct(prior["strict_f1"])
        + r"""へ改善した。改善は自車走行、停止、左折に集中し、映像だけでは曖昧な
camera motionを補助情報で拘束した効果と解釈できる。最終方式"""
        + _tex(final_name)
        + "のF1は"
        + _pct(final["strict_f1"])
        + "、MCC "
        + _number(final["selective_mcc"])
        + "であった。"
        + delta_text
        + r""" 粗いRTK方位を右左折の答えとして直接使わず、同期品質が低い走行は
VOへfallbackしたため、公開派生データの欠損を正解情報で補っていない。"""
    )


def _is_map_context_pass(
    composed_motion: dict[str, Any] | None,
) -> bool:
    if not composed_motion:
        return False
    return set(composed_motion.get("context_tags") or []) == {
        "ego_turn_left",
        "ego_turn_right",
        "traffic_light_red",
        "traffic_light_green",
    }


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
    ci_excludes_zero = float(ci[0]) > 0 or float(ci[1]) < 0
    if delta > 0 and ci_excludes_zero:
        deployment = (
            "品質を優先する場合はcandidateを用い、費用を優先する大量一次"
            "ラベリングにはbaselineを用いる二段構成が妥当である。"
        )
    elif delta > 0:
        deployment = (
            "candidateの点推定値は高いが95\\%信頼区間が0を跨ぐため、"
            "明確な品質優位は確認できない。実装の簡潔性、latency、費用を"
            "含めて運用条件を選ぶ必要がある。"
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


def _public_motion_summary_text(
    motion_prior: dict[str, Any] | None,
    composed_motion: dict[str, Any] | None,
    *,
    baseline_cost: float | None,
) -> dict[str, Any]:
    empty = {
        "name": "",
        "metrics": None,
        "cost": baseline_cost,
        "f1_ci": "算出不能",
        "abstract": "",
        "discussion": "",
        "conclusion": "",
        "recommendation": "",
    }
    if not motion_prior:
        return empty

    baseline = motion_prior["baseline"]["metrics"]
    metrics = motion_prior["fused"]["metrics"]
    name = "Super / 固定VO motion prior"
    bootstrap = motion_prior.get("bootstrap_ci95") or {}
    delta = motion_prior.get("paired_f1_delta") or {}
    extra_cost = 0.0
    composed_is_evaluation = bool(
        composed_motion
        and composed_motion.get("split") == motion_prior.get("split")
    )
    if composed_is_evaluation:
        assert composed_motion is not None
        metrics = composed_motion["composed"]["metrics"]
        name = "Super / VO prior＋車線変更pass"
        bootstrap = (
            composed_motion["composed"].get("bootstrap_ci95") or {}
        )
        delta = (
            composed_motion["composed"].get(
                "paired_f1_delta_vs_baseline"
            )
            or {}
        )
        runtime = composed_motion.get("context_runtime") or {}
        extra_cost = float(
            runtime.get("estimated_usd_per_1000_clips") or 0
        )
    cost = (
        float(baseline_cost) + extra_cost
        if baseline_cost is not None
        else None
    )
    ci = bootstrap.get("strict_f1")
    f1_ci = (
        f"{_pct(ci[0])}--{_pct(ci[1])}" if ci else "算出不能"
    )
    delta_ci = delta.get("ci95")
    delta_ci_text = (
        f"{100 * float(delta_ci[0]):+.1f}--"
        f"{100 * float(delta_ci[1]):+.1f} points"
        if delta_ci
        else "算出不能"
    )
    abstract = (
        "公開ROAD動画から推定したVOを用いる固定motion priorにより、"
        f"SuperのF1を{_pct(baseline['strict_f1'])}から"
        f"{_pct(metrics['strict_f1'])}へ改善した。"
    )
    if composed_is_evaluation:
        abstract += (
            "最終構成は車線変更だけを追加passで再判定し、"
            "他12タグを再生成しない。"
        )
    discussion = (
        "公開motion fusionは、公式生INS/GNSSを使用せず、認証不要の公開派生軌跡と"
        "全動画から再計算可能なVOだけで構成した。"
        f"映像baselineに対するF1差のsource-video block bootstrap 95\\% CIは"
        f"{delta_ci_text}である。motion prior自体は追加GPU requestを要さず、"
        "閾値はdevelopmentで固定してevaluationでは変更していない。"
    )
    conclusion = (
        "さらに、静的背景のvisual flowとyawを用いる固定fusionは、"
        f"Super映像baselineのF1 {_pct(baseline['strict_f1'])}を"
        f"{_pct(metrics['strict_f1'])}へ改善し、"
        "frame追加よりも自車運動の構造化が有効であることを示した。"
    )
    recommendation = (
        f"公開VOの固定motion priorを追加し、F1 {_pct(metrics['strict_f1'])}を得た。"
        "motion priorは追加GPU requestを要しないが、CPUによるVO前処理費用は"
        "GPU推論費用へ含めていない。"
    )
    return {
        "name": name,
        "metrics": metrics,
        "cost": cost,
        "f1_ci": f1_ci,
        "abstract": abstract,
        "discussion": discussion,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }


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
    motion_prior: dict[str, Any] | None = None,
    composed_motion: dict[str, Any] | None = None,
    context_fusion: dict[str, Any] | None = None,
    opr_motion: dict[str, Any] | None = None,
    visual_motion: dict[str, Any] | None = None,
    map_context: dict[str, Any] | None = None,
    prompt_development: dict[str, Any] | None = None,
    prompt_selection: dict[str, Any] | None = None,
    prompt_audit: dict[str, Any] | None = None,
    recall_diagnostic: dict[str, Any] | None = None,
    recall_confidence_baseline: dict[str, Any] | None = None,
    recall_confidence_oracle: dict[str, Any] | None = None,
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
            motion_prior,
            composed_motion,
            context_fusion,
            opr_motion,
            visual_motion,
            map_context,
        )
    )
    lines.extend(
        _prompt_recall_markdown(
            prompt_development,
            prompt_selection,
            prompt_audit,
            recall_diagnostic,
            recall_confidence_baseline,
            recall_confidence_oracle,
        )
    )
    return "\n".join(lines) + "\n"


def _supplemental_markdown(
    error_analysis: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    serving_repeat: dict[str, Any] | None,
    cascade: dict[str, Any] | None,
    motion_prior: dict[str, Any] | None,
    composed_motion: dict[str, Any] | None,
    context_fusion: dict[str, Any] | None,
    opr_motion: dict[str, Any] | None,
    visual_motion: dict[str, Any] | None,
    map_context: dict[str, Any] | None,
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
    if motion_prior:
        baseline = motion_prior["baseline"]["metrics"]
        prior = motion_prior["fused"]["metrics"]
        lines.extend(
            [
                "",
                "## Public Motion Context and Fusion",
                "",
                "公式の生INS/GNSSは使用せず、公開OPR派生軌跡と"
                "ROAD動画から算出したVOだけを使用した。",
                (
                    f"- OPR coverage: {opr_motion['clip_count']} clips / "
                    f"{opr_motion['source_video_count']} source videos"
                    if opr_motion
                    else "- OPR coverage: unavailable"
                ),
                (
                    f"- Video-derived VO: {visual_motion['clip_count']} clips / "
                    f"{visual_motion['source_video_count']} source videos"
                    if visual_motion
                    else "- Video-derived VO: all configured clips"
                ),
                "",
                "| Method | Precision | Recall | F1 | MCC |",
                "|---|---:|---:|---:|---:|",
                f"| Super video baseline | {_pct(baseline['precision'])} | "
                f"{_pct(baseline['strict_recall'])} | "
                f"{_pct(baseline['strict_f1'])} | "
                f"{_number(baseline['selective_mcc'])} |",
                f"| Frozen VO motion prior | {_pct(prior['precision'])} | "
                f"{_pct(prior['strict_recall'])} | "
                f"{_pct(prior['strict_f1'])} | "
                f"{_number(prior['selective_mcc'])} |",
            ]
        )
        if (
            composed_motion
            and composed_motion.get("split") == motion_prior.get("split")
        ):
            composed = composed_motion["composed"]["metrics"]
            lines.append(
                f"| Motion prior + scoped VO pass | "
                f"{_pct(composed['precision'])} | "
                f"{_pct(composed['strict_recall'])} | "
                f"{_pct(composed['strict_f1'])} | "
                f"{_number(composed['selective_mcc'])} |"
            )
        elif composed_motion:
            development_prior = (
                composed_motion.get("motion_prior", {}).get("metrics")
                or prior
            )
            scoped = composed_motion["composed"]["metrics"]
            changes = composed_motion["composed"][
                "paired_changes_vs_motion_prior"
            ]
            runtime = composed_motion.get("context_runtime") or {}
            if _is_map_context_pass(composed_motion):
                snapshot = (map_context or {}).get("osm_snapshot") or {}
                description = (
                    "Development-only OSM/OPR map-context pass was "
                    "rejected: "
                    f"F1 {_pct(development_prior['strict_f1'])} -> "
                    f"{_pct(scoped['strict_f1'])}, "
                    f"{changes['improved_pairs']} improved / "
                    f"{changes['regressed_pairs']} regressed pairs, "
                    "additional cost "
                    f"${_number(runtime.get('estimated_usd_per_1000_clips'), 2)}"
                    " per 1,000 target clips. "
                    f"OSM snapshot SHA-256: {snapshot.get('sha256', 'unknown')}."
                )
            else:
                description = (
                    "Development-only lane-change follow-up was rejected: "
                    f"F1 {_pct(scoped['strict_f1'])}, "
                    f"{changes['improved_pairs']} improved / "
                    f"{changes['regressed_pairs']} regressed pairs."
                )
            lines.extend(
                [
                    "",
                    description,
                ]
            )
    return lines


def _prompt_recall_markdown(
    prompt_development: dict[str, Any] | None,
    prompt_selection: dict[str, Any] | None,
    prompt_audit: dict[str, Any] | None,
    recall_diagnostic: dict[str, Any] | None,
    recall_confidence_baseline: dict[str, Any] | None,
    recall_confidence_oracle: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    if prompt_development and prompt_selection:
        selected_profile = str(prompt_selection["selected"]["profile"])
        preferred = [
            "road-adaptive-24-reasoned",
            "road-adaptive-24-contrastive-v2",
            "road-adaptive-24-hybrid-core-v1",
            "road-adaptive-24-hybrid-temporal-v1",
            "road-adaptive-24-hybrid-f1-v1",
        ]
        lines.extend(
            [
                "",
                "## Development-only Prompt Selection",
                "",
                "| Prompt | Criteria words | Detailed tags | Precision | "
                "Recall | F1 | MCC |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        rows = {
            str(row["profile"]): row
            for row in prompt_development.get("profiles") or []
        }
        for profile in preferred:
            row = rows.get(profile)
            if row is None:
                continue
            label = profile + (" (selected)" if profile == selected_profile else "")
            variant = _profile_to_prompt_variant(profile)
            micro = row["micro"]
            lines.append(
                f"| {label} | {_prompt_audit_words(prompt_audit, variant)} | "
                f"{_prompt_audit_detailed_tags(prompt_audit, variant)} | "
                f"{_pct(micro['precision'])} | "
                f"{_pct(micro['strict_recall'])} | "
                f"{_pct(micro['strict_f1'])} | "
                f"{_number(micro['selective_mcc'])} |"
            )
        lines.extend(
            [
                "",
                "These values select a prompt on the Development split. "
                "They are not Evaluation performance estimates.",
            ]
        )
    if recall_diagnostic:
        lines.extend(
            [
                "",
                "## Development-only Recall Diagnostic",
                "",
                "| Condition | FN recovered | TP retained | New FP | "
                "Frames | E2E | USD/1k proxy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in recall_diagnostic.get("profiles") or []:
            lines.append(
                f"| {row['profile']} | "
                f"{row['original_false_negatives_recovered']}/"
                f"{row['original_false_negative_count']} | "
                f"{row['original_true_positives_retained']}/"
                f"{row['original_true_positive_count']} | "
                f"{row['negative_control_false_positives']}/"
                f"{row['negative_control_count']} | "
                f"{row['mean_selected_frames']:.1f} | "
                f"{row['mean_e2e_seconds']:.2f} | "
                f"{row['estimated_usd_per_1000_proxy']:.2f} |"
            )
        lines.extend(
            [
                "",
                "GT event times and GT boxes are diagnostic-only upper bounds. "
                "The set intentionally over-samples difficult errors.",
            ]
        )
    confidence_summaries = [
        ("Baseline 24", recall_confidence_baseline),
        ("Oracle 48", recall_confidence_oracle),
    ]
    if any(summary for _, summary in confidence_summaries):
        lines.extend(
            [
                "",
                "## Exploratory Tag-specific Confidence",
                "",
                "| Media | Threshold | FN recovered | TP retained | "
                "New FP | Precision | Recall |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for media, summary in confidence_summaries:
            if not summary:
                continue
            for row in summary.get("confidence", {}).get("thresholds") or []:
                if round(float(row["threshold"]), 2) not in {
                    0.05,
                    0.20,
                    0.40,
                }:
                    continue
                lines.append(
                    f"| {media} | {float(row['threshold']):.2f} | "
                    f"{row['original_false_negatives_recovered']}/14 | "
                    f"{row['original_true_positives_retained']}/7 | "
                    f"{row['negative_control_false_positives']}/7 | "
                    f"{_pct(row['precision'])} | {_pct(row['recall'])} |"
                )
        lines.extend(
            [
                "",
                "Thresholds are exploratory and require an independent "
                "calibration split before production use.",
            ]
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
PATH=/opt/homebrew/bin:$PATH PYTHONPATH=benchmark \\
  python3 -m benchmark_tool.road_paper_cli --paper-dir paper
```

PDF generation requires Tectonic and the fonts declared in the TeX source.
The artifact manifest records SHA-256 digests for every input and output.
"""


def _render_adversarial_self_review() -> str:
    return """# 敵対的自己査読記録

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
