### 1 · What a "sample" is

CICIDS2017 rows are already-aggregated **flows**, not packets. One flow on its own
is a summary of a session. A window here is **16 consecutive flows arriving at one
monitored host, on one day** — the view an operator has of a server they own.

Two rules make that window safe:

* **Grouped by host.** Sliding over the globally time-sorted file would interleave
  unrelated machines, and the recurrence would learn "what time of day is it".
* **Never spanning a day.** The evaluation split is by weekday; a window that
  straddled Wednesday into Thursday would put the same flows on both sides.

### 2 · Which columns are allowed

Dropped, always: `Flow ID`, source and destination IP, source port, timestamp.
In this capture almost every attack comes from `172.16.0.1`, so an IP column *is*
the label. Timestamp and source IP are still used — to group and order flows —
but never as inputs.

Dropped by default: `Destination Port`, which is a near-label here (port 21/22
traffic is essentially the brute-force attack). It can be switched back on, and
the report then carries both numbers.

Kept as-is: duplicate rows. De-duplicating is standard advice for the
*row-classification* framing, and wrong here — a brute force **is** a run of
near-identical flows.

### 3 · The Elman network

```
h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

The hidden state of the previous step is fed back as extra input — Elman's
"context units". A Jordan network feeds back the *output* instead. Keras'
`SimpleRNN` is exactly the Elman recurrence.

```
Input (16, 78) → SimpleRNN(64, tanh) → Dropout(0.2) → Dense(1, sigmoid)
```

### 4 · What it is measured against

Every rival sees the same window of the same features at a matched parameter
budget. Only the ability to use order changes:

| model | sees the window? | can use order? |
|---|---|---|
| Elman | yes | yes, recurrently |
| window MLP | yes | positionally, after flattening |
| window MLP (wide) | yes | same, ~9× the parameters |
| last-flow MLP | only the final flow | no |
| logistic regression | yes, flattened | positionally |

### 5 · The three controls

* **Within-host.** Pooled scores can come from recognising *which machine* sent a
  window. Inside one host that is constant, so within-host numbers mean detection.
* **Order shuffle.** Permuting the flows inside every window keeps the contents
  and destroys the order. No drop means the recurrence was decoration.
* **The instrument check.** Before believing any null result, the harness is run
  on a probe where order is the *only* signal. An earlier configuration scored
  chance on it — and would have produced a confident, false "recurrence doesn't
  help". The current one reaches PR-AUC 0.985 there.
