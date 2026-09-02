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
        "footer": "ElmanGuard · licença MIT · uso em pesquisa apenas",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    value = TEXT.get(lang, TEXT["en"]).get(key, TEXT["en"].get(key, key))
    return value.format(**kwargs) if kwargs else value
