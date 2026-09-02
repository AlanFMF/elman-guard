### 1 · O que é uma "amostra"

As linhas do CICIDS2017 já são **fluxos** agregados, não pacotes. Um fluxo
sozinho é o resumo de uma sessão. Aqui, uma janela são **16 fluxos consecutivos
chegando em um host monitorado, em um mesmo dia** — a visão que um operador tem
de um servidor seu.

Duas regras tornam essa janela segura:

* **Agrupada por host.** Deslizar sobre o arquivo ordenado globalmente
  intercalaria máquinas sem relação, e a recorrência aprenderia "que horas são".
* **Nunca atravessa um dia.** A divisão de avaliação é por dia da semana; uma
  janela que fosse de quarta para quinta colocaria os mesmos fluxos dos dois lados.

### 2 · Quais colunas são permitidas

Sempre descartadas: `Flow ID`, IPs de origem e destino, porta de origem,
timestamp. Nesta captura quase todo ataque parte de `172.16.0.1`, então uma
coluna de IP **é** o rótulo. Timestamp e IP de origem continuam sendo usados —
para agrupar e ordenar os fluxos — mas nunca como entrada.

Descartada por padrão: `Destination Port`, que aqui é quase um rótulo (tráfego
nas portas 21/22 é essencialmente o ataque de força bruta). Dá para reativar, e
o relatório então traz os dois números.

Mantidas: linhas duplicadas. Remover duplicatas é o conselho padrão para o
enquadramento de *classificação de linhas*, e está errado aqui — uma força bruta
**é** uma sequência de fluxos quase idênticos.

### 3 · A rede de Elman

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

O estado oculto do passo anterior é realimentado como entrada extra — as
"unidades de contexto" de Elman. A rede de Jordan realimenta a *saída*. O
`SimpleRNN` do Keras é exatamente a recorrência de Elman.

```
Entrada (16, 78) → SimpleRNN(64, tanh) → Dropout(0,2) → Dense(1, sigmoid)
```

### 4 · Contra o que ela é medida

Todo rival vê a mesma janela, com as mesmas features e orçamento de parâmetros
equivalente. Só muda a capacidade de usar a ordem:

| modelo | vê a janela? | pode usar a ordem? |
|---|---|---|
| Elman | sim | sim, de forma recorrente |
| MLP da janela | sim | posicionalmente, após achatar |
| MLP da janela (larga) | sim | igual, com ~9× os parâmetros |
| MLP do último fluxo | só o fluxo final | não |
| regressão logística | sim, achatada | posicionalmente |

### 5 · Os três controles

* **Dentro do host.** A pontuação agregada pode vir de reconhecer *qual máquina*
  enviou a janela. Dentro de um host isso é constante, então os números por host
  significam detecção.
* **Embaralhamento da ordem.** Permutar os fluxos dentro de cada janela preserva
  o conteúdo e destrói a ordem. Nenhuma queda significa que a recorrência era
  enfeite.
* **Aferição do instrumento.** Antes de acreditar em qualquer resultado nulo, o
  harness é rodado em uma sonda onde a ordem é o *único* sinal. Uma configuração
  anterior ficou no acaso ali — e teria produzido um "a recorrência não ajuda"
  confiante e falso. A atual chega a PR-AUC 0,985 nessa sonda.
