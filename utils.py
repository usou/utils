#!/usr/bin/env python
# coding: utf-8

import inspect
from pathlib import Path
import numpy as np
import json


def get_args_of_current_function(n_skip_args=None):
    """親の関数の引数を取得する

    see https://tottoto.net/python3-get-args-of-current-function/

    Parameters
    ----------
    n_skip_args: int
        最初から何個の引数を無視するか。selfやclsなどを除去するため。

    Returns
    -------
    args: dict
        {arg_key: arg_value}

    Examples
    --------
    >>> def f(a, b):
    ...     return get_args_of_current_function(n_skip_args=1)
    >>> f(a=1, b=2)
    {'b': 2}
    """
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    args = {key: info.locals[key] for key in info.args[n_skip_args:]}
    return args


def numbering_daily_dir(root_log_dir, today, n_max_dirs=999, mkdir_ok=False):
    """日付ごとに1から(デフォルトで)999まで連番を振る

    Parameters
    ----------
    root_log_dir: str
        directory name
    today: datetime
        datetime型。datetime.datetime.now()などを渡しても、日付未満を除去。
    n_max_dirs: int
        作成する最大のディレクトリ数。ディレクトリ名の長さに影響する。
    mkdir_ok: bool
        root_log_dirが存在していなかったとき、新たに作成する

    Returns
    -------
    dir_: str
        日付+連番。例）2020/1/31の場合, "200131001"から振られる

    Examples
    --------
    >>> import tempfile
    >>> import datetime
    >>> with tempfile.TemporaryDirectory() as dname:
    ...      today = datetime.datetime(2020,1,31)
    ...      numbering_daily_dir(dname, today)
    '200131001'
    """
    p = Path(root_log_dir)
    if (not p.exists()) and mkdir_ok:
        p.mkdir()
    if (not p.exists()) and (not mkdir_ok):
        raise ValueError("'{}' does not exist. Make 'mkdir_ok' option True".format(root_log_dir))
    dirs = p.iterdir()

    # --- 日付+連番の後ろに名前が入っている場合は除去
    n_digits = len(str(n_max_dirs))
    # 日付が6文字、連番がn_digits文字
    length_dir = 6 + n_digits
    dirs = [dir_.name[:length_dir] for dir_ in dirs]

    today = today.strftime("%y%m%d")
    for i in range(1, n_max_dirs+1):
        dir_ = today+"{num:0{n_digits}}".format(num=i, n_digits=n_digits)
        if dir_ not in dirs:
            return dir_


def split_data_with(a, propotion, seed=0):
    """aの要素によってデータをN分割する。propotionでNとその割合を指定する。

    Parameters
    ----------
    a: array_like
        複数のユニークな要素を持つarray.
        このarrayと同じサイズのboolsがlen(propotion)だけ作られる
    propotion: array_like
        要素ごとに分割する割合を指定する。
        ユニークな要素に関して割合を指定していることに注意。
        (aのそれぞれ要素が均一でない限り、boolsの割合とここで指定した割合は一致しない）

    Returns
    -------
    bools: tupple
        N個のnp.arrayのboolsのタプル

    Examples
    --------
    >>> a = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

    >>> propotion = [.8, .2]
    >>> bools_tr, bools_te = split_data_with(a, propotion, seed=0)
    >>> a[bools_tr], a[bools_te]
    (array([2, 2, 3, 3, 3, 4, 4, 4, 4]), array([1]))

    >>> propotion = [.8, .2]
    >>> bools_tr, bools_te = split_data_with(a, propotion, seed=5)
    >>> a[bools_tr], a[bools_te]
    (array([1, 2, 2, 3, 3, 3]), array([4, 4, 4, 4]))

    >>> propotion = [.6, .2, .2]
    >>> bools_tr, bools_va, bools_te = split_data_with(a, propotion, seed=0)
    >>> a[bools_tr], a[bools_va], a[bools_te]
    (array([3, 3, 3, 4, 4, 4, 4]), array([2, 2]), array([1]))
    """
    if sum(propotion) != 1.:
        raise ValueError("The sum of propotion must be 1.")

    np.random.seed(seed)
    uniques = np.unique(a)
    uniques = np.random.permutation(uniques)
    n_uniques = len(uniques)

    if n_uniques < len(propotion):
        raise ValueError("n_propotions must be equal or smaller than n_uniques.\n \
                n_propotions: {}, n_uniques: {}".format(len(propotion), n_uniques))

    # --- 各割合に割り当てられるunique valuesを選ぶ
    uniques_list = []
    props = [0] + np.cumsum(propotion).tolist()
    for i in range(len(props)-1):
        start = int(n_uniques*props[i])
        end   = int(n_uniques*props[i+1])
        uniques_list.append(uniques[start:end])

    bools = tuple(np.isin(a, uniques) for uniques in uniques_list)
    return bools


def create_my_json_encoder(encoded_class, encoder):
    """encoder methodを追加した新しいjson encoder classを作成する

    Parameters
    ----------
    encoded_class: class
        エンコードされるclassを指定
    encoder: function
        encodeする処理を記述

    Returns
    -------
    MyJSONEncoder:
        json.JSONEncoderを親クラスに持つEncoder

    Examples
    --------
    >>> import tempfile
    >>> data = np.array(5)
    >>> with tempfile.TemporaryFile(mode="w") as f:
    ...     json.dump(data, f, cls=create_my_json_encoder(np.ndarray, int))
    """
    class MyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, encoded_class):
                return encoder(obj)
            return super().default(obj)
    return MyJSONEncoder


if __name__ == "__main__":
    import doctest
    doctest.testmod()
