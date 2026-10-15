"""Bilingual strings for the ElmanGuard app (English / Brazilian Portuguese)."""

from __future__ import annotations

LANGUAGES = {"English": "en", "Português (BR)": "pt"}

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "ElmanGuard",
        "tagline": "Sequence-based intrusion detection with a classic Elman recurrent network — "
                   "and the controls that say whether it is really detecting anything.",
        "disclaimer_title": "Research demonstration",
        "disclaimer": (
            "Trained on CICIDS2017, a 2017 lab capture. It is a study of evaluation "
            "methodology, **not** a production IDS: the traffic, the attacks and the network "
            "layout are all synthetic scenarios from one week in one testbed."
        ),
        "sidebar_data": "1 · Traffic",
        "source": "What should I look at?",
        "source_bundled": "Bundled sample (test day)",
        "source_upload": "Upload flows (CSV)",
        "source_synthetic": "Synthetic traffic",
        "host_pick": "Monitored host",
        "day_pick": "Day",
        "sidebar_view": "2 · View",
        "threshold": "Alert threshold (false-positive budget)",
        "model_pick": "Model",
        "timeline_title": "Attack probability over the day",
        "timeline_help": (
            "Each point is a window of 16 consecutive flows arriving at this host. "
            "The shaded bands are windows the ground truth marks as attack."
        ),
        "metric_windows": "Windows",
        "metric_attack": "Attack windows (truth)",
        "metric_flagged": "Flagged by the model",
        "metric_attack_delta": "{pct} of windows",
        "p_attack": "P(attack)",
        "download": "Download CSV",
        "window_table": "Window table",
        "metric_recall": "Recall at this threshold",
        "results_title": "Measured results",
        "results_intro": (
            "From the chronological split: trained Monday–Wednesday, threshold set on Thursday, "
            "evaluated on Friday. Every attack type in the test day is one the model has "
            "**never seen**, which is the situation a deployed detector is actually in."
        ),
        "control_title": "The control that matters",
        "control_intro": (
            "Pooled across hosts, a model can score well by recognising *which machine* a window "
            "came from: in this capture the attacked server is nearly always under attack and the "
            "others nearly never are. Inside a single host that information is constant, so the "
            "within-host numbers are the ones that mean detection rather than recognition."
        ),
        "ablation_title": "Does the recurrence use time order?",
        "ablation_intro": (
            "Shuffling the flows inside every window keeps the contents and destroys the order. "
            "A model that scores the same either way was reading the window as an unordered bag."
        ),
        "how_title": "How it works",
        "no_reports": "Run `python scripts/run_experiment.py` to generate the measured results.",
        "prevalence_floor_label": "prevalence floor",
        "protocols_title": "How much does the evaluation protocol change the answer?",
        "protocols_intro": (
            "The same models on the same windows, scored three ways. **Random** shuffles windows "
            "into train and test — the protocol most published CICIDS2017 results use. "
            "**Temporal** cuts every attack burst in time and drops any window that shares a flow "
            "across the boundary: the attack types are known, the flows are not. "
            "**Chronological** trains Monday–Wednesday and tests Friday: every attack type is new. "
            "The first gap is leakage; the second is the cost of meeting an attack for the first time."
        ),
        "protocol_names": "random|temporal, embargoed|chronological by day",
        "shared_flows": "test windows sharing a training flow",
        "mc_title": "Which attack is it? (multiclass)",
        "mc_intro": (
            "Detection asks whether something is wrong; the operator's next question is *what*. "
            "Scored by **macro-F1**, where every attack type counts the same — a model cannot score "
            "well by being right about benign traffic and DoS Hulk alone. Uses the temporal split: "
            "on the chronological one the test day only holds types never seen in training, and a "
            "classifier cannot name a class it was never shown."
        ),
        "mc_excluded": "Excluded as too rare to learn or score (windows in the whole capture): {items}.",
        "mc_cm_caption": "Elman network, first seed. Each row is a true attack type and sums to 100%.",
        "mc_ablation": "Does the order of flows help **name** the attack?",
        "footer": "ElmanGuard · MIT licence · research use only",
    },
    "pt": {
        "title": "ElmanGuard",
        "tagline": "Detecção de intrusões por sequências com uma rede recorrente de Elman — "
                   "e os controles que dizem se ela está mesmo detectando alguma coisa.",
        "disclaimer_title": "Demonstração de pesquisa",
        "disclaimer": (
            "Treinado no CICIDS2017, uma captura de laboratório de 2017. É um estudo de "
            "metodologia de avaliação, **não** um IDS de produção: o tráfego, os ataques e a "
            "topologia são cenários sintéticos de uma semana em um único testbed."
        ),
        "sidebar_data": "1 · Tráfego",
        "source": "O que eu devo olhar?",
        "source_bundled": "Amostra incluída (dia de teste)",
        "source_upload": "Enviar fluxos (CSV)",
        "source_synthetic": "Tráfego sintético",
        "host_pick": "Host monitorado",
        "day_pick": "Dia",
        "sidebar_view": "2 · Visualização",
        "threshold": "Limiar de alerta (orçamento de falsos positivos)",
        "model_pick": "Modelo",
        "timeline_title": "Probabilidade de ataque ao longo do dia",
        "timeline_help": (
            "Cada ponto é uma janela de 16 fluxos consecutivos chegando neste host. "
            "As faixas sombreadas são janelas que o rótulo verdadeiro marca como ataque."
        ),
        "metric_windows": "Janelas",
        "metric_attack": "Janelas de ataque (real)",
        "metric_flagged": "Sinalizadas pelo modelo",
        "metric_attack_delta": "{pct} das janelas",
        "p_attack": "P(ataque)",
        "download": "Baixar CSV",
        "window_table": "Tabela de janelas",
        "metric_recall": "Recall neste limiar",
        "results_title": "Resultados medidos",
        "results_intro": (
            "Do split cronológico: treino de segunda a quarta, limiar definido na quinta, "
            "avaliação na sexta. Todo tipo de ataque do dia de teste é um que o modelo "
            "**nunca viu** — que é a situação real de um detector em produção."
        ),
        "control_title": "O controle que importa",
        "control_intro": (
            "Agregado entre hosts, um modelo pode pontuar bem apenas reconhecendo *qual máquina* "
            "originou a janela: nesta captura o servidor atacado está quase sempre sob ataque e os "
            "demais quase nunca. Dentro de um mesmo host essa informação é constante, então os "
            "números por host são os que significam detecção, e não reconhecimento."
        ),
        "ablation_title": "A recorrência usa a ordem temporal?",
        "ablation_intro": (
            "Embaralhar os fluxos dentro de cada janela preserva o conteúdo e destrói a ordem. "
            "Um modelo que pontua igual dos dois jeitos estava lendo a janela como um saco "
            "desordenado de fluxos."
        ),
        "how_title": "Como funciona",
        "no_reports": "Rode `python scripts/run_experiment.py` para gerar os resultados medidos.",
        "prevalence_floor_label": "piso de prevalência",
        "protocols_title": "Quanto o protocolo de avaliação muda a resposta?",
        "protocols_intro": (
            "Os mesmos modelos, nas mesmas janelas, avaliados de três jeitos. O **aleatório** "
            "embaralha as janelas entre treino e teste — o protocolo da maioria dos resultados "
            "publicados com o CICIDS2017. O **temporal** corta cada rajada de ataque no tempo e "
            "descarta qualquer janela que compartilhe um fluxo através da fronteira: os tipos de "
            "ataque são conhecidos, os fluxos não. O **cronológico** treina de segunda a quarta e "
            "testa na sexta: todo tipo de ataque é novo. A primeira diferença é vazamento; a segunda "
            "é o custo de encontrar um ataque pela primeira vez."
        ),
        "protocol_names": "aleatório|temporal com embargo|cronológico por dia",
        "shared_flows": "janelas de teste com algum fluxo do treino",
        "mc_title": "Qual ataque é este? (multiclasse)",
        "mc_intro": (
            "A detecção pergunta se há algo errado; a próxima pergunta do operador é *o quê*. "
            "Avaliado por **F1 macro**, em que todo tipo de ataque pesa igual — um modelo não "
            "pontua bem acertando só o tráfego benigno e o DoS Hulk. Usa o split temporal: no "
            "cronológico, o dia de teste só tem tipos nunca vistos no treino, e um classificador "
            "não consegue nomear uma classe que nunca lhe foi mostrada."
        ),
        "mc_excluded": "Excluídos por serem raros demais para aprender ou avaliar (janelas na captura inteira): {items}.",
        "mc_cm_caption": "Rede de Elman, primeira semente. Cada linha é um tipo de ataque real e soma 100%.",
        "mc_ablation": "A ordem dos fluxos ajuda a **nomear** o ataque?",
        "footer": "ElmanGuard · licença MIT · uso em pesquisa apenas",
    },
}


MODEL_NAMES: dict[str, dict[str, str]] = {
    "en": {
        "elman": "Elman RNN", "window_mlp": "window MLP",
        "window_mlp_wide": "window MLP (wide)", "last_flow_mlp": "last-flow MLP",
        "logistic": "logistic regression", "prevalence_floor": "prevalence floor",
    },
    "pt": {
        "elman": "Rede de Elman", "window_mlp": "MLP da janela",
        "window_mlp_wide": "MLP da janela (larga)", "last_flow_mlp": "MLP do último fluxo",
        "logistic": "regressão logística", "prevalence_floor": "piso de prevalência",
    },
}

COLUMN_NAMES: dict[str, dict[str, str]] = {
    "en": {
        "pr_auc_mean": "PR-AUC", "pr_auc_std": "sd (3 seeds)", "roc_auc_mean": "ROC-AUC",
        "recall_mean": "recall @ 1% FPR", "precision_mean": "precision", "f1_mean": "F1",
        "model": "model", "attack type": "attack type", "host": "host",
        "n_windows": "windows", "positive_rate": "attack rate", "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC", "recall": "recall", "fpr": "FPR",
        "pr_auc_ordered": "PR-AUC ordered", "pr_auc_shuffled": "PR-AUC shuffled",
        "pr_auc_drop": "drop", "recall_drop": "recall drop",
        "pr_auc_ordered_sd": "sd ordered", "pr_auc_shuffled_sd": "sd shuffled",
        "macro_f1_mean": "macro-F1", "macro_f1_std": "sd (3 seeds)",
        "balanced_accuracy_mean": "balanced accuracy", "balanced_accuracy_std": "sd",
        "weighted_f1_mean": "weighted F1", "weighted_f1_std": "sd ",
        "macro_f1_ordered": "macro-F1 ordered", "macro_f1_ordered_sd": "sd ordered",
        "macro_f1_shuffled": "macro-F1 shuffled", "macro_f1_shuffled_sd": "sd shuffled",
        "macro_f1_drop": "drop", "train": "train", "calib": "calibration", "test": "test",
        "in_task": "in the task",
    },
    "pt": {
        "pr_auc_mean": "PR-AUC", "pr_auc_std": "desvio (3 sementes)", "roc_auc_mean": "ROC-AUC",
        "recall_mean": "recall a 1% FPR", "precision_mean": "precisão", "f1_mean": "F1",
        "model": "modelo", "attack type": "tipo de ataque", "host": "host",
        "n_windows": "janelas", "positive_rate": "taxa de ataque", "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC", "recall": "recall", "fpr": "FPR",
        "pr_auc_ordered": "PR-AUC ordenado", "pr_auc_shuffled": "PR-AUC embaralhado",
        "pr_auc_drop": "queda", "recall_drop": "queda no recall",
        "pr_auc_ordered_sd": "desvio ordenado", "pr_auc_shuffled_sd": "desvio embaralhado",
        "macro_f1_mean": "F1 macro", "macro_f1_std": "desvio (3 sementes)",
        "balanced_accuracy_mean": "acurácia balanceada", "balanced_accuracy_std": "desvio",
        "weighted_f1_mean": "F1 ponderado", "weighted_f1_std": "desvio ",
        "macro_f1_ordered": "F1 macro ordenado", "macro_f1_ordered_sd": "desvio ordenado",
        "macro_f1_shuffled": "F1 macro embaralhado", "macro_f1_shuffled_sd": "desvio embaralhado",
        "macro_f1_drop": "queda", "train": "treino", "calib": "calibração", "test": "teste",
        "in_task": "na tarefa",
    },
}


def pretty(df, lang: str):
    """Rename a results frame's index and columns for display."""
    models = MODEL_NAMES.get(lang, MODEL_NAMES["en"])
    cols = COLUMN_NAMES.get(lang, COLUMN_NAMES["en"])
    # models appear as the index in some frames and as columns in others
    out = df.rename(columns={**cols, **models})
    out = out.rename(index=models)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: models.get(v, v) if isinstance(v, str) else v)
    out.index.name = cols.get(df.index.name, df.index.name)
    return out


def t(lang: str, key: str, **kwargs) -> str:
    value = TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))
    return value.format(**kwargs) if kwargs else value
