<h1 align="center">ElmanGuard</h1>

<p align="center">
  <strong>Detecção de intrusões por sequências com uma rede recorrente de Elman — e os controles que dizem se ela está mesmo detectando alguma coisa.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.21-ff6f00?logo=tensorflow&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-ff4b4b?logo=streamlit&logoColor=white">
  <img alt="Dataset" src="https://img.shields.io/badge/dataset-CICIDS2017-1baf7a">
  <img alt="License" src="https://img.shields.io/badge/licença-MIT-2a78d6">
</p>

<p align="center"><a href="README.en.md">🇺🇸 Read in English</a></p>

---

Uma rede de Elman lê uma janela de 16 fluxos de rede consecutivos chegando em um
servidor monitorado e prevê se aquela janela contém um ataque. Treinada de
segunda a quarta do CICIDS2017, com limiar definido na quinta e avaliada na
sexta — cujos tipos de ataque ela nunca viu.

O ponto do repositório não é o detector. São os quatro controles em volta dele, e
o que eles revelam.

## Três achados

**1 · A ordem temporal não carrega sinal mensurável aqui — e esse nulo é crível.**
Embaralhar os fluxos dentro de cada janela destrói a ordem e preserva o conteúdo.
O PR-AUC do Elman cai **0,055**; o desvio-padrão da mesma medida entre sementes é
**0,117**. O efeito é menor que o ruído. O que torna isso digno de ser afirmado é
a aferição do instrumento, abaixo: o mesmo harness detecta um sinal temporal
*plantado* com PR-AUC 0,985.

**2 · A maior parte do desempenho aparente é reconhecimento de host, não detecção de ataque.**
Agregados entre servidores, os modelos chegam a PR-AUC 0,31–0,55. Calculados
*dentro* de cada host — onde "qual máquina é esta" não carrega informação —
todos os modelos neurais ficam no acaso ou abaixo dele, em todos os hosts. Só uma
regressão logística no único servidor pesadamente atacado supera o acaso, com
ROC-AUC 0,694.

**3 · Um modelo linear vence todas as arquiteturas neurais, e as quatro redes são
indistinguíveis entre si.**

![Comparação de arquiteturas](assets/architecture_comparison.png)

<details>
<summary><strong>O app</strong> (clique para expandir)</summary>

![App Streamlit](assets/app_screenshot.png)

O dia de teste no servidor atacado. O dia inteiro está sombreado porque 98,6% das
suas janelas contêm tráfego de ataque — e a probabilidade do modelo passa longos
trechos perto de zero lá dentro. Essa imagem é o controle por host, desenhado.

</details>

## O experimento

| | |
|---|---|
| Dados | CICIDS2017 `TrafficLabelling`, os 8 arquivos, **2.827.876 fluxos** após reparo |
| Features | **78** estatísticas numéricas de fluxo; IPs, portas, `Flow ID` e timestamp excluídos |
| Amostra | 16 fluxos consecutivos chegando em um host monitorado, em um mesmo dia |
| Divisão | treino seg–qua (82.968 janelas, 3,2% ataque) · limiar qui (29.955, 1,1%) · teste sex (23.858, 8,1%) |
| Ataques do teste | PortScan, DDoS, Bot — **nenhum deles visto no treino** |
| Repetições | todo número abaixo é média ± desvio sobre 3 sementes |

## Arquitetura

O caminho que um fluxo de rede percorre até virar um número deste README:

```mermaid
flowchart TB
    CSV["8 CSVs do CICIDS2017<br/>2.827.876 fluxos"] --> Rep["Reparo de 6 defeitos<br/>nomes de coluna, latin-1,<br/>timestamps de 12h sem AM/PM"]
    Rep --> Feat["78 features numéricas<br/>IPs, portas e Flow ID ficam de fora"]
    Feat --> Win["Janelas de 16 fluxos consecutivos<br/>por host monitorado, por dia"]

    Win --> Tr["Treino: seg a qua<br/>82.968 janelas"]
    Win --> Ca["Calibração: quinta<br/>define o limiar de alerta"]
    Win --> Te["Teste: sexta<br/>ataques nunca vistos no treino"]

    Tr --> Mod
    Ca --> Mod
    Te --> Mod

    Mod["Elman · MLP da janela · MLP larga<br/>MLP do último fluxo · logística<br/>orçamento de parâmetros equivalente"] --> Res["Resultado agregado<br/>3 sementes, média e desvio"]

    Res --> C1["Controle 1<br/>métricas dentro de cada host"]
    Res --> C2["Controle 2<br/>ordem embaralhada na janela"]
    Res --> C3["Controle 3<br/>aferição do instrumento"]
```

| Etapa | O que acontece | Por que importa |
|---|---|---|
| Reparo | Seis defeitos dos CSVs publicados são corrigidos e testados | Sem isso o arquivo nem carrega direito, e dois deles corrompem o tempo |
| Features | IPs, portas, `Flow ID` e timestamp saem da matriz | Nesta captura o IP do atacante **é** o rótulo |
| Sequências | 16 fluxos consecutivos do mesmo host, no mesmo dia | Janela por host evita que a rede aprenda "que horas são"; nunca cruzar o dia evita vazamento no split |
| Divisão | Cronológica, por dia da semana | Cada dia tem ataques diferentes, então o teste mede generalização para tipos inéditos |
| Modelos | Cinco, com orçamento de parâmetros equivalente | Isola arquitetura de capacidade |
| Controles | Três, aplicados sobre o resultado | É o que separa detecção de reconhecimento, e sinal de ruído |

### A rede de Elman

A camada de contexto guarda a ativação da camada oculta do passo anterior e a
devolve como entrada extra. A rede de Jordan, sua irmã, realimenta a camada de
saída — a diferença está no ponto de onde parte a seta pontilhada:

```mermaid
flowchart LR
    X["x_t<br/>features do fluxo no passo t"] --> H["Camada oculta<br/>h_t = tanh(Wx·x_t + Wh·h_t-1 + b)"]
    H --> Y["Saída<br/>P(ataque na janela)"]
    H --> Ctx["Camada de contexto<br/>guarda h_t-1"]
    Ctx -. "realimenta a entrada oculta" .-> H
    Y -. "a rede de Jordan realimentaria daqui" .-> Ctx
```

### O atalho que os controles procuram

Quase todo ataque do CICIDS2017 parte de um único host e atinge um único
servidor. Isso abre um caminho pelo qual um modelo pontua bem sem detectar nada:

```mermaid
flowchart TB
    Q["Uma janela de 16 fluxos<br/>chegando em um servidor"] --> A{"O modelo acertou porque leu o tráfego<br/>ou porque reconheceu o host?"}
    A -->|"medindo todos os hosts juntos"| P["PR-AUC de 0,31 a 0,55<br/>parece detecção"]
    A -->|"medindo dentro de cada host"| W["ROC-AUC no acaso<br/>em quase todos os servidores"]
    P --> D["A distância entre os dois números<br/>é o tamanho do atalho de identidade"]
    W --> D
```

## Resultados

PR-AUC no dia de teste com ataques inéditos. O piso de prevalência — o que um
detector que alerta em tudo obteria — é 0,081.

| modelo | PR-AUC | ROC-AUC | recall a 1% FPR | precisão | params |
|---|---|---|---|---|---|
| **regressão logística** | **0,546** | 0,775 | **0,468** | 0,825 | 19.969 |
| MLP da janela (larga) | 0,349 ± 0,008 | 0,601 | 0,258 ± 0,019 | 0,822 | 80.001 |
| MLP da janela | 0,347 ± 0,050 | 0,603 | 0,155 ± 0,124 | 0,841 | 8.751 |
| **Elman RNN** | 0,341 ± 0,117 | 0,638 | 0,177 ± 0,097 | 0,815 | 9.217 |
| MLP do último fluxo | 0,309 ± 0,020 | 0,758 | 0,174 ± 0,056 | 0,767 | 9.201 |
| piso de prevalência | 0,081 | 0,500 | — | 0,081 | — |

Os quatro modelos neurais vão de 0,309 a 0,349 — uma faixa mais estreita que o
próprio desvio-padrão do Elman entre sementes. **Qualquer artigo que reporte que
sua RNN vence um MLP por dois pontos no CICIDS2017 está reportando a semente.**

O recall por tipo de ataque mostra de onde vem o 0,55:

| ataque (inédito) | Elman | MLP da janela | logística |
|---|---|---|---|
| DDoS | 0,530 | 0,684 | 0,808 |
| PortScan | 0,000 | 0,000 | 0,431 |
| Bot | 0,000 | 0,000 | 0,036 |

O DDoS transfere porque os dias de treino são cheios de DoS. PortScan e Bot não
transferem nada para os modelos neurais. É assim que "generalizar para tipos de
ataque inéditos" se parece de verdade.

## Os controles

### 1 · Dentro do host: quanto é reconhecimento?

Agregado entre servidores, um modelo pode pontuar bem apenas aprendendo *quais
máquinas são atacadas* — nesta captura o servidor web está sob ataque em 95% da
semana e os demais praticamente nunca. Dentro de um host essa informação é
constante, então os números por host medem detecção.

ROC-AUC, dia de teste, por host monitorado (0,5 = acaso):

| host | taxa de ataque | Elman | MLP da janela | logística |
|---|---|---|---|---|
| 192.168.10.50 | 98,6% | 0,561 | 0,451 | **0,694** |
| 192.168.10.9 | 13,8% | 0,426 | 0,399 | 0,509 |
| 192.168.10.15 | 8,0% | 0,483 | 0,340 | 0,328 |
| 192.168.10.5 | 4,9% | 0,417 | 0,360 | 0,415 |
| 192.168.10.14 | 4,3% | 0,316 | 0,192 | 0,303 |
| 192.168.10.8 | 3,5% | 0,464 | 0,266 | 0,341 |
| 205.174.165.73 | 56,9% | 0,220 | 0,100 | 0,195 |

Lendo a tabela: **o PR-AUC agregado de 0,34–0,55 é em grande parte identidade de
host.** Tirando isso, uma única célula supera o acaso.

### 2 · Embaralhamento da ordem: a recorrência se paga?

| modelo | PR-AUC ordenado | PR-AUC embaralhado | queda | desvio entre sementes |
|---|---|---|---|---|
| Elman RNN | 0,341 ± 0,117 | 0,286 ± 0,039 | 0,055 | 0,117 |
| MLP da janela | 0,347 ± 0,050 | 0,305 ± 0,024 | 0,042 | 0,050 |

As duas quedas são menores que a dispersão entre execuções. No nível de janela de
fluxos desta captura, a ordem não carrega informação que os modelos consigam usar.

### 3 · A aferição do instrumento — o controle que salvou o projeto

Um resultado nulo não vale nada se o instrumento não é capaz de detectar um
positivo. O `synthetic.order_only_dataset` monta janelas em que **a ordem é o
único sinal**: toda janela vem do mesmo processo AR(1), e as negativas são os
mesmos valores permutados no tempo — então média, variância, mínimo, máximo e até
a distribuição do último passo são idênticas entre as classes.

| configuração | PR-AUC na sonda de ordem pura |
|---|---|
| padrão anterior (48 units, dropout 0,3, lr 1e-3) | **0,483** — acaso |
| padrão atual (64 units, dropout 0,2, lr 3e-3) | **0,985** |

A primeira configuração teria produzido uma conclusão confiante, publicada e
completamente falsa: *"a recorrência não acrescenta nada"*. Não era um achado,
era uma rede subtreinada. A sonda hoje é um teste
(`tests/test_harness_detects_planted_signal.py`) que falha se o harness perder a
capacidade de enxergar um efeito temporal.

Um segundo bug de reprodutibilidade apareceu do mesmo jeito: semear dentro do
`fit`, mas não antes de os pesos serem criados, fazia duas execuções da mesma
configuração diferirem em mais de 0,25 de PR-AUC. Daí três sementes em todo
número.

### 4 · Agrupar por origem em vez de destino

Quase todo fluxo de ataque do CICIDS2017 parte de um host, `172.16.0.1`, cujo
tráfego é **99,4% ataque**. Agrupar as sequências por origem transforma a pergunta,
silenciosamente, em "esta é a máquina do atacante?". Medido, execução única:

| agrupamento | PR-AUC | precisão |
|---|---|---|
| por destino (servidor monitorado) | 0,470 | 0,815 |
| por origem | 0,291 | **1,000** |

Eu esperava que agrupar por origem inflasse o PR-AUC, e não inflou; o que ele faz
é empurrar a precisão para 1,000 exatos, que é a assinatura de um modelo que
achou um atalho de identidade e só dispara quando tem certeza. Reportado como
medido, não como previsto.

## Método

Detalhe completo em [`docs/how_it_works.pt.md`](docs/how_it_works.pt.md). Em resumo:

**O parser corrige seis defeitos documentados** dos CSVs publicados — espaços à
esquerda em 70 dos 85 nomes de coluna, bytes latin-1 nos rótulos de web attack,
linhas de cabeçalho no meio do arquivo, 288.602 linhas vazias de preenchimento,
timestamps de 12 horas sem marcador AM/PM (`1:02` é 13:02) e ordem dia/mês
ambígua, resolvida conferindo o dia da semana contra o nome do arquivo. Cada um
tem teste.

**Linhas duplicadas são mantidas, de propósito.** Remover duplicatas é o conselho
padrão e vem do enquadramento de classificação de linhas. Aqui a divisão é
cronológica, então duplicatas não atravessam a fronteira — e uma força bruta *é*
uma sequência de fluxos quase idênticos. Removê-las apagaria o sinal. A contagem
é reportada em vez disso.

**Janelas nunca cruzam host nem dia**, então o split por dia da semana não pode
colocar os mesmos fluxos dos dois lados.

**Toda arquitetura recebe orçamento de parâmetros equivalente**, porque uma janela
achatada de 16 passos é 16× mais larga que um passo e uma camada densa sem esse
ajuste venceria por capacidade, não por arquitetura. O `window_mlp_wide` é o
controle deliberadamente superdimensionado, com 9× os parâmetros.

**O early stopping observa janelas retidas dos dias de treino**, não o dia de
calibração. Cada dia da semana tem ataques diferentes, então um sinal de validação
de outro dia mede transferência, não convergência, e o treino para no ruído.

```
Entrada (16, 78) → SimpleRNN(64, tanh) → Dropout(0,2) → Dense(1, sigmoid)
9.217 parâmetros · entropia cruzada binária · Adam 3e-3 · com peso de classe
```

## Como rodar

O CICIDS2017 não é redistribuível, então a captura bruta não está no repositório.

```bash
git clone https://github.com/AlanFMF/elman-guard.git
cd elman-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -m "not slow"        # 22 testes, sem precisar do dataset
pytest                      # inclui as aferições do instrumento (treina modelos pequenos)
```

Com os dados (baixe o `GeneratedLabelledFlows.zip` do
[CIC](https://www.unb.ca/cic/datasets/ids-2017.html) — **não** a versão
`MachineLearningCVE`, que teve as colunas de IP e timestamp removidas e não
permite formar sequências):

```bash
python scripts/prepare_data.py --raw caminho/para/TrafficLabelling --out data/processed
python scripts/run_experiment.py --split chronological --seeds 0 1 2
python scripts/run_experiment.py --split random --seeds 0 1 2   # o protocolo inflado
python scripts/export_demo.py
python scripts/make_figures.py
streamlit run app.py
```

O app é bilíngue e mostra a probabilidade de ataque ao longo do dia de teste para
um servidor escolhido, com os períodos de ataque reais sombreados, ao lado de
todos os resultados medidos acima.

## Limitações

- **Uma semana, um testbed, 2017.** O tráfego é gerado, os ataques são
  roteirizados e a topologia é fixa. Nada aqui transfere para uma rede real sem
  nova medição.
- **Defeitos conhecidos do dataset.** Além dos seis que o parser corrige, há
  trabalho publicado documentando erros de rotulagem e um bug de construção de
  fluxos no CICFlowMeter usado para gerar estes CSVs. Existe uma re-publicação
  corrigida; este projeto usa a original, então os números absolutos carregam essa
  ressalva.
- **A janela tem 16 fluxos.** Um ataque lento espalhado por horas pode
  simplesmente não caber no contexto. Janelas maiores não foram varridas.
- **Três sementes** bastam para mostrar que as diferenças são ruído, mas não para
  resolver diferenças reais pequenas.
- **Só os protocolos cronológico e aleatório** foram comparados. Divisão disjunta
  por host é impossível aqui: uma única máquina envia quase todo o tráfego de
  ataque.

### O que eu tentaria a seguir

Janelas mais longas e multiescala; um modelo de linha de base por host, para que
a pontuação seja "incomum *para este servidor*" em vez de "incomum"; e a
re-publicação corrigida do CICIDS2017, para ver quais destes achados sobrevivem a
rótulos melhores.

## ⚠️ Aviso

Demonstração de pesquisa sobre metodologia de avaliação de modelos sequenciais,
**não** um sistema de detecção de intrusões de produção. Não implante em rede real.

## Licença

MIT — veja [LICENSE](LICENSE).
