# -*- coding: utf-8 -*-

# cfg, tomlだとdict of dictや
#複数のパラメータをリストで渡す、などやりづらいので
#global変数でparamsを渡してしまう

# ------------------------------------------------------------
# output_dirだけはAggregateMetricsのclass parameterに直接渡す必要があります。。。。
# ------------------------------------------------------------

my_params =\
    [
        {"seed": seed,
         "data": dict(path=dict(X="../../../data/processed/dpf/us.npy",
                                Y="../../../data/processed/dpf/Y.pkl"),
                      label="temp",
                      propotion=[.6, .2, .2],
                      splited_with="date",
                      metric = "r2_score",
                      ),
         "model": dict(cls_name = "sklearn.linear_model.Ridge",
                       params   = dict(alpha=1.),
                      ),
         }
     for seed in range(3)
     ]
