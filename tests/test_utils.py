import signal
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import yaml
from pytest import mark, raises

from adb_graphics import conversions, utils


@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def test_cfgrib_spec_no_model():
    config = {"foo": {"bar": "baz"}}
    answer = utils.cfgrib_spec(config, "model")
    assert answer == config


def test_cfgrib_spec_model():
    config = {"model": {"foo": {"bar": "baz"}}}
    answer = utils.cfgrib_spec(config, "model")
    assert answer == config["model"]


def test_create_zip(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.touch()
    bfile.touch()
    zipf = tmp_path / "file.zip"
    utils.create_zip([str(f) for f in [afile, bfile]], zipf)
    assert zipf.is_file()
    assert not afile.is_file()
    assert not bfile.is_file()


def test_create_zip_locked(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.touch()
    bfile.touch()
    zipf = tmp_path / "file.zip"
    zipf_lock = tmp_path / "file.zip._lock"
    zipf_lock.touch()
    with raises(TimeoutError), timeout(3):
        utils.create_zip([str(f) for f in [afile, bfile]], zipf)
    assert not zipf.is_file()
    assert afile.is_file()
    assert bfile.is_file()


@mark.parametrize(
    ("arg", "expected"),
    [
        ([1], [1]),
        ([1, 9], list(range(1, 10))),
        ([1, 9, 3], list(range(1, 10, 3))),
        ([3, 4, 7, 19], [3, 4, 7, 19]),
    ],
)
def test_fhr_list(arg, expected):
    assert utils.fhr_list(arg) == expected


@mark.parametrize(
    ("arg", "expected"),
    [
        (datetime(2025, 10, 31, 12), "2025103112"),
        (datetime(2025, 10, 31), "2025103100"),
        (datetime(2025, 10, 31, 12, 1, 2), "2025103112"),
    ],
)
def test_from_datetime(arg, expected):
    assert utils.from_datetime(arg) == expected


@mark.parametrize(
    ("arg", "expected"),
    [
        ("conversions.to_micro", conversions.to_micro),
        ("utils.join_ranges", utils.join_ranges),
    ],
)
def test_get_func(arg, expected):
    assert utils.get_func(arg) == expected


def test_get_func_undefined():
    with raises(ValueError):  # noqa: PT011
        utils.get_func("foo.bar")


def test_join_ranges():
    yaml_str = """
    a: !join_ranges [[0, 10, 0.1], [10, 51, 1.0]]
    b: !join_ranges [[0, 5], [4]]
    c: !join_ranges [[2, 17, 7]]
    """
    yaml.add_constructor("!join_ranges", utils.join_ranges, Loader=yaml.SafeLoader)

    d = yaml.safe_load(yaml_str)
    assert np.array_equal(d["a"], np.concatenate([np.arange(0, 10, 0.1), np.arange(10, 51, 1.0)]))
    assert np.array_equal(d["b"], np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3]))
    assert np.array_equal(d["c"], np.asarray([2, 9, 16]))


def test_arange_constructor():
    yaml_str = """
    a: !arange [0, 10, 0.1]
    b: !arange [0, 5]
    c: !arange [2, 17, 7]
    """
    yaml.add_constructor("!arange", utils.arange_constructor, Loader=yaml.SafeLoader)

    d = yaml.safe_load(yaml_str)
    assert np.array_equal(d["a"], np.arange(0, 10, 0.1))
    assert np.array_equal(d["b"], np.asarray([0, 1, 2, 3, 4]))
    assert np.array_equal(d["c"], np.asarray([2, 9, 16]))


def test_load_yaml(tmp_path):
    yaml_str = """
    a: !float '{{ c[1] - 2 }}'
    b: !join_ranges [[0, 5], [4]]
    c: !arange [2, 17, 7]
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_str)
    d = utils.load_yaml(cfg)
    d.dereference()
    assert d["a"] == 7
    assert np.array_equal(d["b"], np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3]))
    assert np.array_equal(d["c"], np.asarray([2, 9, 16]))
