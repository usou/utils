# -*- coding: utf-8 -*-

import luigi
from luigi.parameter import _DictParamEncoder
import csv
import pickle
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import shutil
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import sklearn

import sys
import pathlib
parent_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path += [str(parent_dir)]
from utils import split_data_with

def copy_file(file_):
    def _copy_file(func):
        def wrapper(*args, **kargs):
            self = args[0]
            p = Path(self.output_dir)
            if not p.exists():
                p.mkdir()
            shutil.copyfile(file_, str(p/file_))

            return func(*args, **kargs)
        return wrapper
    return _copy_file


class UtilsMixin:
    def get_params_dict(self):
        names = self.get_param_names()
        params = {name: getattr(self, name) for name in names}
        return params

    def _dict2json(self, dict_, file_obj):
        with file_obj as fout:
            # FrozenOrderedDict to Dict
            json.dump(dict_, fout, cls=_DictParamEncoder)

    def _obj2pickle(self, obj, file_obj):
        with file_obj as fout:
            pickle.dump(obj, fout)

    def _dict2csv(self, dict_, file_obj):
        with file_obj as fout:
            w = csv.DictWriter(fout, dict_.keys())
            w.writeheader()
            w.writerow(dict_)
    def _df2csv(self, df, file_obj):
        with self.output().open('w') as fout:
            df.to_csv(fout, index=False)


class BaseAggregateMetrics(luigi.Task, UtilsMixin, metaclass=ABCMeta):
    @abstractmethod
    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_dir, "metrics.csv"))

    def run(self):
        df = pd.DataFrame()
        for task in luigi.task.flatten(self.requires()):
            with task.output()['params'].open() as f:
                df_params = pd.io.json.json_normalize(json.load(f))

            with task.output()['scores'].open() as f:
                df_scores = pd.read_csv(f)

            df_con = pd.concat([df_params, df_scores], axis=1)
            df = df.append(df_con, ignore_index=True)

        self._df2csv(df, self.output().open('w'))


class BaseLearn(luigi.Task, UtilsMixin):
    def output(self):
        base_path = os.path.join(self.output_dir, self.task_id) + "/"
        return dict(params=luigi.LocalTarget(path=base_path+'params.json'),
                    model =luigi.LocalTarget(path=base_path+'model.pkl', format=luigi.format.Nop),
                    scores=luigi.LocalTarget(path=base_path+'scores.csv'),
                    )

    def run(self):
        np.random.seed(self.seed)

        params = self.get_params_dict()
        self._dict2json(params, self.output()['params'].open("w"))

        self.load_data()

        self.train()
        self._obj2pickle(self._model, self.output()['model'].open("wb"))

        scores = self.evaluate(self.data["metric"])
        self._dict2csv(scores, self.output()['scores'].open("w"))


    def load_data(self):
        # X: array-like, Y: pd.DataFrame
        X = np.load(self.data["path"]["X"])
        Y = np.load(self.data["path"]["Y"])

        X, Y = self._query_data(X, Y)
        X, y = np.array(X), np.array(Y[self.data["label"]])

        # --- split data
        bools, data_names = self._get_bools(getattr(Y, self.data["splited_with"]))
        self.data_names = data_names
        for bools_, data_name in zip(bools, data_names):
            setattr(self, "X_"+data_name, X[bools_])
            setattr(self, "y_"+data_name, y[bools_])

    def _query_data(self, X, Y):
        bools = ~pd.isnull(Y[self.data["label"]])
        X = X[bools]
        Y = Y[bools]
        return X, Y

    def _get_bools(self, a):
        bools = split_data_with(a, propotion=self.data["propotion"], seed=self.seed)
        # bools is a tuple of (tr, te) or (tr, va, te) bools.
        if len(bools) == 2:
            data_names = ("tr", "te")
        elif len(bools) == 3:
            data_names = ("tr", "va", "te")
        return bools, data_names

    def train(self):
        def _get_learner(cls_name, model_params):
            model_cls = eval(cls_name)
            model = model_cls(**model_params)
            return model

        model = _get_learner(self.model["cls_name"], self.model["params"])
        model.fit(self.X_tr, self.y_tr)
        self._model = model

    def evaluate(self, metric):
        scores = OrderedDict()
        for data_name in self.data_names:
            X      = getattr(self, "X_"+data_name)
            y_true = getattr(self, "y_"+data_name)
            y_pred = self._model.predict(X)
            score = getattr(sklearn.metrics, metric)(y_true, y_pred)
            scores["score_"+data_name] = score

        return scores
