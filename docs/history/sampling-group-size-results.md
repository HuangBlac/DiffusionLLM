# Fixed checkpoint sampling comparison

> 历史实验／过程档案：正文状态、预算与操作指令仅对应文中日期，不作为现行执行依据。2026-11-06整理后的当前状态见[项目进度](../progress.md)，P2执行约束见[P2规程](../current/p2-readiness-2026-11-05.md)。原始测量和失败证据保留。


Inference only on the existing 200-step checkpoints; no training or Colab compute. Twelve prompts were fixed before inspecting outputs; the first three are the historical prompts. Both methods greedily reveal the most confident masked positions. Only group size changes: 1 versus 8 tokens. Each output has 32 tokens, with all special tokens (including EOS) suppressed, matching the old fixed-length protocol.

These are token groups, not necessarily words. This diagnostic cannot establish natural termination, general language quality, or whether more training would help. Repetition metrics are not coherence scores. Greedy timing repeats are identical generations, not independent quality samples.

## Aggregate measurements

| Checkpoint | Tokens per round | Forwards/output | Mean repeated 4-grams | Mean adjacent repetition | Mean dominant token share | Median per-prompt latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shift | 1 | 32 | 69.0% | 38.4% | 52.9% | 2.415 |
| shift | 8 | 4 | 82.2% | 91.7% | 87.0% | 0.317 |
| same | 1 | 32 | 78.7% | 48.1% | 67.7% | 2.420 |
| same | 8 | 4 | 82.5% | 92.2% | 87.2% | 0.303 |

Latency includes tokenization, per-round trace extraction and synchronization in this diagnostic implementation; excludes model loading and warmup. It is not an optimized serving benchmark or a compute-matched quality comparison. The one-token condition uses 8x as many model forwards.

## All outputs

### shift checkpoint

#### 1. The purpose of science is

1 token(s) per round:

```text
 to understand the the nature of the universe, the , , , , , , , , , , , , , , , , , , the , , ,
```

8 token(s) per round:

```text
 to to to to to , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 2. Once upon a time, a small bird

1 token(s) per round:

```text
 was seen by a passerian , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

8 token(s) per round:

```text
 was was a , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 3. A language model learns to

1 token(s) per round:

```text
 recognize the and and and and the and and the and and the and and and and and and and the and and and and the and the and the and and
```

8 token(s) per round:

```text
 recognize the the language , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 4. The city was founded in

1 token(s) per round:

```text
 1800 , 1800 , 1800 , 1800 , 1800 , 1
```

8 token(s) per round:

```text
      , , ,  , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 5. Water is essential for life because

1 token(s) per round:

```text
 of the presence of the presence of the presence of the presence of the presence of the presence of the presence of the presence of the presence of the presence of the
```

8 token(s) per round:

```text
 of the water and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

#### 6. The researchers compared the two methods and found that

1 token(s) per round:

```text
 the two methods were similar in terms of the accuracy of the two methods. The researchers found that the two methods were similar in terms of the accuracy of the two
```

8 token(s) per round:

```text
 the and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

#### 7. Although the experiment was carefully designed,

1 token(s) per round:

```text
 the results were not statistically significant, the results were not statistically significant, the results were not statistically significant, the results were not statistically significant, the results were not
```

8 token(s) per round:

```text
 the the results and and and the the were were , , , , , , , , , , , , , , , , , , , , , ,
```

#### 8. The main difference between a river and a lake is

1 token(s) per round:

```text
 the amount of water in the lake. The main difference between a river and a lake is the amount of water in the lake. The main difference between a river
```

8 token(s) per round:

```text
 the the the water , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 9. After the rain stopped, the children

1 token(s) per round:

```text
 were playing and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

8 token(s) per round:

```text
 were the children and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

#### 10. To understand the history of this region,

1 token(s) per round:

```text
 it is necessary to understand the political and economic and economic and economic and economic and economic and economic and economic and economic and economic and economic and economic and economic and
```

8 token(s) per round:

```text
 it the the the region of , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 11. The book describes how

1 token(s) per round:

```text
 the and and and and and and and and and and and and the and and and and and and and and and and the and and and and and and the
```

8 token(s) per round:

```text
 the book and and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

#### 12. When the temperature falls below freezing,

1 token(s) per round:

```text
 the air can become extremely dry and extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely extremely
```

8 token(s) per round:

```text
 the the temperature , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

### same checkpoint

#### 1. The purpose of science is

1 token(s) per round:

```text
 to the the of of the of the of the of the of the of the of the of the of the of the of the of the of the of the
```

8 token(s) per round:

```text
 to to the , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 2. Once upon a time, a small bird

1 token(s) per round:

```text
 was , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

8 token(s) per round:

```text
 was caught , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 3. A language model learns to

1 token(s) per round:

```text
 understand the of of of the language and the of the of the and the of the and the of the and the of and the of and the of and and
```

8 token(s) per round:

```text
 understand the the . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
```

#### 4. The city was founded in

1 token(s) per round:

```text
 1999 and 1999 and 1999 and 1999 and 1999 and 1
```

8 token(s) per round:

```text
 1  , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 5. Water is essential for life because

1 token(s) per round:

```text
 it is a source of food and . . . . . . . . . . . . . . . . . . . . . . . . .
```

8 token(s) per round:

```text
 it is the , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 6. The researchers compared the two methods and found that

1 token(s) per round:

```text
 the and were and and and and and and and and and and and and and and and and and and and and and and and and and and and and and
```

8 token(s) per round:

```text
 the and and and were and and and and and , and , , , , , , , , , , , , , , , , , , , ,
```

#### 7. Although the experiment was carefully designed,

1 token(s) per round:

```text
 the of of of the was . . The . The of the was . . . . . . . . . . . . . . . . . .
```

8 token(s) per round:

```text
 the the the , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 8. The main difference between a river and a lake is

1 token(s) per round:

```text
 the of of of the of the of the of the of the of the of the of the of the of the of the of the of the of the of
```

8 token(s) per round:

```text
 the the the , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 9. After the rain stopped, the children

1 token(s) per round:

```text
 were , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

8 token(s) per round:

```text
 were , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

#### 10. To understand the history of this region,

1 token(s) per round:

```text
 the of of of the of the of the of the of the of the of the of the of the of the of the of the of the of the of
```

8 token(s) per round:

```text
 the the the the the , , , , , , , , , , , , , , , ,  , , ,  ,     
```

#### 11. The book describes how

1 token(s) per round:

```text
 the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of
```

8 token(s) per round:

```text
 the the the the                            
```

#### 12. When the temperature falls below freezing,

1 token(s) per round:

```text
 the air is , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
```

8 token(s) per round:

```text
 the                               
```

## Provenance and scope

Both checkpoint hashes were checked against their original training records. Exact restoration of every adapter tensor and reproduction of all three historical samples were verified before measurement. The new decoder also reproduced the old four-token-per-round outputs. All parameters were frozen, no optimizer was created, and unchanged checkpoint/adapter values were verified after inference. Raw records contain token IDs, reveal order, confidence and all timing repeats.

[Raw results](../../results/sampling-group-size-200.json).
