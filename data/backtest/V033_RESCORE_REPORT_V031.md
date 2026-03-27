# Backtest Rescore Report — V031

**Weights version:** v031
**Backtest file:** C:\Users\Unknown\Downloads\mlb-prop-predictor\data\backtest\backtest_2025_v032_old.json
**Tradeable props only:** batter_strikeouts, earned_runs, hits, hits_runs_rbis, hitter_fantasy_score, pitcher_strikeouts, pitching_outs, rbis, runs, total_bases, walks_allowed

## Summary: All Picks vs Selected Picks

| Prop                           | Dir   | Floor |   All N | All Acc |   Sel N | Sel Acc |
|--------------------------------|-------|-------|---------|---------|---------|---------|
| batter_strikeouts              | MORE  | 0.66 |   36259 |   63.3% |    6587 |   71.1% |
| batter_strikeouts              | LESS  | 0.66 |    4087 |   56.1% |       0 |     n/a |
| earned_runs                    | MORE  | 0.60 |    4020 |   61.8% |      52 |   61.5% |
| earned_runs                    | LESS  | 0.60 |      54 |   46.3% |       0 |     n/a |
| hits                           | LESS  | 0.72 |   40346 |   78.2% |   30077 |   79.3% |
| hits_runs_rbis                 | MORE  | 0.95 |   14546 |   51.5% |       0 |     n/a |
| hits_runs_rbis                 | LESS  | 0.95 |   25800 |   57.5% |       0 |     n/a |
| hitter_fantasy_score           | MORE  | 0.56 |     495 |   55.2% |      42 |   57.1% |
| hitter_fantasy_score           | LESS  | 0.68 |   39851 |   65.7% |       1 |  100.0% |
| pitcher_strikeouts             | MORE  | 0.66 |    2739 |   58.3% |      76 |   78.9% |
| pitcher_strikeouts             | LESS  | 0.66 |    1335 |   58.9% |      11 |   63.6% |
| pitching_outs                  | LESS  | 0.60 |    4074 |   59.5% |     197 |   73.6% |
| rbis                           | MORE  | 0.60 |     204 |   46.1% |       0 |     n/a |
| rbis                           | LESS  | 0.60 |   40142 |   70.3% |   35159 |   71.2% |
| runs                           | MORE  | 0.68 |      30 |   60.0% |       0 |     n/a |
| runs                           | LESS  | 0.68 |   40316 |   62.4% |       0 |     n/a |
| total_bases                    | MORE  | 0.64 |    1227 |   47.8% |       6 |   50.0% |
| total_bases                    | LESS  | 0.72 |   39119 |   65.0% |       0 |     n/a |
| walks_allowed                  | MORE  | 0.95 |    3340 |   52.8% |       0 |     n/a |
| walks_allowed                  | LESS  | 0.95 |     734 |   55.7% |       0 |     n/a |
|--------------------------------|-------|-------|---------|---------|---------|---------|
| TOTAL (tradeable)              |       | 0.00 |  298718 |   65.1% |   72208 |   74.6% |

## Direction Bias Analysis (MORE vs LESS)

| Prop                           | MORE Sel N |  MORE Acc | LESS Sel N |  LESS Acc |  Gap (L-M) |
|--------------------------------|------------|-----------|------------|-----------|------------|
| batter_strikeouts              |       6587 |     71.1% |          0 |       n/a |        n/a |
| earned_runs                    |         52 |     61.5% |          0 |       n/a |        n/a |
| hits                           |          0 |       n/a |      30077 |     79.3% |        n/a |
| hitter_fantasy_score           |         42 |     57.1% |          1 |    100.0% |    +42.9pp |
| pitcher_strikeouts             |         76 |     78.9% |         11 |     63.6% |    -15.3pp |
| pitching_outs                  |          0 |       n/a |        197 |     73.6% |        n/a |
| rbis                           |          0 |       n/a |      35159 |     71.2% |        n/a |
| total_bases                    |          6 |     50.0% |          0 |       n/a |        n/a |

## Confidence Bucket Breakdown (Selected Props)

### batter_strikeouts MORE  (floor=0.66)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.70-0.75    |    272 |    200 |   73.5% |
| 0.65-0.70    |   9158 |   6434 |   70.3% |
| 0.60-0.65    |  10282 |   6727 |   65.4% |
| 0.55-0.60    |   7970 |   4924 |   61.8% |
| <0.55        |   8577 |   4664 |   54.4% |

### earned_runs MORE  (floor=0.60)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.60-0.65    |     52 |     32 |   61.5% |
| 0.55-0.60    |   1267 |    801 |   63.2% |
| <0.55        |   2701 |   1652 |   61.2% |

### hits LESS  (floor=0.72)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.75-0.80    |  21450 |  17279 |   80.6% |
| 0.70-0.75    |  12823 |   9742 |   76.0% |
| 0.65-0.70    |   5381 |   4047 |   75.2% |
| 0.60-0.65    |    645 |    459 |   71.2% |
| 0.55-0.60    |     44 |     28 |   63.6% |
| <0.55        |      3 |      2 |   66.7% |

### hitter_fantasy_score MORE  (floor=0.56)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.65-0.70    |      1 |      0 |    0.0% |
| 0.60-0.65    |     25 |     11 |   44.0% |
| 0.55-0.60    |     18 |     15 |   83.3% |
| <0.55        |    451 |    247 |   54.8% |

### hitter_fantasy_score LESS  (floor=0.68)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.65-0.70    |    615 |    471 |   76.6% |
| 0.60-0.65    |  10055 |   7107 |   70.7% |
| 0.55-0.60    |  18186 |  11960 |   65.8% |
| <0.55        |  10995 |   6649 |   60.5% |

### pitcher_strikeouts MORE  (floor=0.66)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.70-0.75    |     34 |     30 |   88.2% |
| 0.65-0.70    |     59 |     40 |   67.8% |
| 0.60-0.65    |     98 |     61 |   62.2% |
| 0.55-0.60    |    432 |    310 |   71.8% |
| <0.55        |   2116 |   1157 |   54.7% |

### pitcher_strikeouts LESS  (floor=0.66)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.65-0.70    |     33 |     29 |   87.9% |
| 0.60-0.65    |    169 |    120 |   71.0% |
| 0.55-0.60    |    171 |    108 |   63.2% |
| <0.55        |    962 |    529 |   55.0% |

### pitching_outs LESS  (floor=0.60)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.75-0.80    |      6 |      6 |  100.0% |
| 0.70-0.75    |     11 |     11 |  100.0% |
| 0.65-0.70    |     82 |     65 |   79.3% |
| 0.60-0.65    |     98 |     63 |   64.3% |
| 0.55-0.60    |    436 |    270 |   61.9% |
| <0.55        |   3441 |   2009 |   58.4% |

### rbis LESS  (floor=0.60)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.65-0.70    |  24993 |  18023 |   72.1% |
| 0.60-0.65    |  10166 |   7015 |   69.0% |
| 0.55-0.60    |   3452 |   2240 |   64.9% |
| <0.55        |   1531 |    950 |   62.1% |

### total_bases MORE  (floor=0.64)

| Bucket       |      N |   Wins |     Acc |
|--------------|--------|--------|---------|
| 0.70-0.75    |      1 |      1 |  100.0% |
| 0.65-0.70    |      5 |      2 |   40.0% |
| 0.60-0.65    |      8 |      3 |   37.5% |
| 0.55-0.60    |     70 |     36 |   51.4% |
| <0.55        |   1143 |    545 |   47.7% |
