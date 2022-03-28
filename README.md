# aquacrop-gym
Environment to train and compare irrigation scheduling strategies with AquaCrop-OSPy

## Environments and benchmarks


### **Nebraska - Maize**

`env_config = nebraska_maize_config`


| Agent type | Optimizer/DRL algo | Observation set | Action set | Train profit | Test profit | Code link |
| --- | --- | ----------- |--- | ----------- |--- | ----------- |
| Fixed soil-moisture thresholds | Differential evolution | default | smt4`*` | 577 | 581 |
| DRL | PPO | default | smt4 | 570 | 564 | fcnet=[256]*2, 7 day decision, time=20mins, eps=10k, logdir=PPO_CropEnv_2022-02-28_16-20-26f7wnoze2 | na |
| DRL | PPO | default | smt4 | 572 | 571 | fcnet=[256]*2, 3 day decision, time=40mins, eps=10k, logdir=PPO_CropEnv_2022-02-28_16-20-26f7wnoze2 | na |



`*` fixed smt: [0.69892958, 0.56825608, 0.35286359, 0.11124398]

### **Nebraska - Maize (no rain)**

`env_config = nebraska_maize_config`


| Agent type | Optimizer/DRL algo | Observation set | Action set | Train profit | Test profit | Code link |
| --- | --- | ----------- |--- | ----------- |--- | ----------- |
| Fixed soil-moisture thresholds | Differential evolution | default | smt4`*` | 356 | 362 |
| DRL | PPO | default | smt4 | 378 | 385 | fcnet=[256]*2, 7 day decision, time=20mins, eps=10k, logdir=PPO_CropEnv_2022-02-28_16-20-26f7wnoze2 | na |







### **Cordoba - Cotton**

`env_config = cordoba_cotton_config`

no fixed cost so using gross margin

| Agent type | Optimizer/DRL algo | Observation set | Action set | Train profit | Test profit | additional info | Code link |
| --- | --- | ----------- |--- | ----------- |--- | ----------- | --- |
| Fixed soil-moisture thresholds | Differential evolution | default | smt4`*` | 1687 | 1694.0 | na | na |
| DRL | PPO | default | smt4 | 1738.1 | 1735.6 | fcnet=[256]*2, 7 day decision, time=20mins, eps=10k, logdir=PPO_CropEnv_2022-02-28_15-50-23digh0g8t | na |


`*` fixed smt: [0.70684434, 0.35135113, 0.25957379, 0.24270739]

### **Cordoba - Cotton (no rain)**

`env_config = cordoba_cotton_config`

no fixed cost so using gross margin

| Agent type | Optimizer/DRL algo | Observation set | Action set | Train profit | Test profit | additional info | Code link |
| --- | --- | ----------- |--- | ----------- |--- | ----------- | --- |
| Fixed soil-moisture thresholds | Differential evolution | default | smt4`*` | 1667.7 | 1663.2 | na | na |
| DRL | PPO | default | smt4 | 1738.1 | 1735.6 | fcnet=[256]*2, 7 day decision, time=20mins, eps=10k, logdir=PPO_CropEnv_2022-02-28_15-50-23digh0g8t | na |


`*` fixed smt: [0.70684434, 0.35135113, 0.25957379, 0.24270739]
