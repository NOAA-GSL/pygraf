import signal
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from os import utime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

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
    utils.create_zip([afile, bfile], zipf)
    with ZipFile(zipf, "r") as zf:
        assert zf.namelist() == ["a.txt", "b.txt"]
    assert not afile.is_file()
    assert not bfile.is_file()

def test_create_zip_existing_empty(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.write_text("foo")
    bfile.write_text("bar")
    zipf = tmp_path / "file.zip"
    zipf.touch()
    assert zipf.stat().st_size == 0
    utils.create_zip([afile, bfile], zipf)
    assert zipf.stat().st_size > 0
    with ZipFile(zipf, "r") as zf:
        assert zf.namelist() == ["a.txt", "b.txt"]
    assert not afile.is_file()
    assert not bfile.is_file()

def test_create_zip_existing_nonempty(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.write_text("foo")
    a_mod_time = datetime(2025, 1, 1, 1, 0, 0).timestamp()
    utime(afile, (a_mod_time, a_mod_time))
    bfile.write_text("bar")
    zipf = tmp_path / "file.zip"
    with ZipFile(zipf, "w") as zf:
        zf.write(afile, arcname=afile.name)
    utils.create_zip([afile, bfile], zipf)
    with ZipFile(zipf, "r") as zf:
        assert zf.namelist() == ["a.txt", "b.txt"]
        # Make sure the file has the older modify time.
        assert datetime(*zf.getinfo("a.txt").date_time) == datetime.fromtimestamp(a_mod_time)
    assert not afile.is_file()
    assert not bfile.is_file()
    # Call again and make sure that the "overwrite" branch is not executed.
    with patch.object(utils, "ZipFile") as zf:
        utils.create_zip([afile, bfile], zipf)
    zf.assert_called_once_with(zipf, "a")



def test_create_zip_existing_nonempty_overwrite(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.write_text("foo")
    bfile.write_text("bar")
    zipf = tmp_path / "file.zip"
    with ZipFile(zipf, "w") as zf:
        zf.write(afile, arcname=afile.name)
    # A newer archive file (mod time > previously archived file) will overwrite an older one.
    a_mod_time = (datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=5)).timestamp()
    utime(afile, (a_mod_time, a_mod_time))
    utils.create_zip([afile, bfile], zipf)
    with ZipFile(zipf, "r") as zf:
        assert zf.namelist() == ["a.txt", "b.txt"]
        # Make sure the archived file has the newer time.
        assert datetime(*zf.getinfo("a.txt").date_time) == datetime.fromtimestamp(a_mod_time)
    assert not afile.is_file()
    assert not bfile.is_file()


def test_create_zip_error(tmp_path):
    zipf = tmp_path / "file.zip"
    # Using a different error here (not Exception or RuntimeError) to make sure anything gets
    # caught in code under test.
    with (
        patch.object(utils.ZipFile, "write", side_effect=ValueError) as run,
        raises(RuntimeError, match="Error writing zip file!"),
    ):
        utils.create_zip([Path(f) for f in ("afile", "bfile")], zipf)
    assert run.call_count == 2


def test_create_zip_locked(tmp_path):
    afile = tmp_path / "a.txt"
    bfile = tmp_path / "b.txt"
    afile.touch()
    bfile.touch()
    zipf = tmp_path / "file.zip"
    zipf_lock = tmp_path / "file.zip._lock"
    zipf_lock.touch()
    with raises(TimeoutError), timeout(2):
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


def test_load_sites():
    sites_file = Path(__name__).parent.parent / "static" / "conus_raobs.txt"
    sites = utils.load_sites(sites_file)
    assert len(sites) == 91


def test_load_sites_dne():
    sites_file = Path("foo.txt")
    with raises(FileNotFoundError):
        utils.load_sites(sites_file)


def test_load_sites_str():
    sites_file = Path(__name__).parent.parent / "static" / "conus_raobs.txt"
    sites = utils.load_sites(str(sites_file))
    assert len(sites) == 91


def test_load_specs():
    specs_file = Path(__name__).parent.resolve() / "adb_graphics" / "default_specs.yml"
    specs = utils.load_specs(specs_file)
    assert specs["file"] == specs_file


def test_load_specs_dne():
    specs_file = Path("foo.txt")
    with raises(FileNotFoundError) as e:
        utils.load_specs(specs_file)
    assert str(specs_file) in str(e.value)


def test_load_specs_str():
    specs_file = Path(__name__).parent.resolve() / "adb_graphics" / "default_specs.yml"
    specs = utils.load_specs(str(specs_file))
    assert specs["file"] == specs_file


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


@mark.parametrize(
    ("lev", "expected"),
    [
        ("max", ("", "")),
        ("mup", ("", "")),
        ("sfc", ("", "")),
        ("mx02", (2, "mx")),
        ("06km", (6, "km")),
        ("100mb", (100, "mb")),
        ("320m", (320, "m")),
        ("6000ft", (6000, "ft")),
    ],
)
def test_numeric_level(expected, lev):
    assert utils.numeric_level(lev) == expected


@mark.parametrize("age", [0, 1, -1])
def test_old_enough(age, tmp_path):
    path = tmp_path / "foo.txt"
    path.touch()
    old_enough = utils.old_enough(age, path)
    if age < 1:
        assert old_enough
    else:
        assert not old_enough


def test_path_exists():
    path = Path(__name__).parent.resolve()
    assert utils.path_exists(path) == path


def test_path_exists_dne():
    path = Path("foo.txt")
    with raises(FileNotFoundError) as e:
        utils.path_exists(path)
    assert str(path) in str(e.value)


def test_path_exists_str():
    path = Path(__name__).parent.resolve()
    assert utils.path_exists(str(path)) == path


@mark.parametrize(
    "spec",
    [
        {"level": 1},
        {"topLevel": 200},
        {"model": {"bottomLevel": 1}},
        {"Surface": 29},
    ],
)
def test_set_level_nlevel(spec):
    orig = deepcopy(spec)
    utils.set_level(level="200mb", model="model", spec={"cfgrib": spec})
    assert spec.get("level") == orig.get("level")


@mark.parametrize(
    ("level", "expected"),
    [
        ("100mb", 100),
        ("600m", 600),
        ("10m", 10),
    ],
)
def test_set_level_nlevel_no_level_info(expected, level):
    spec: dict = {}
    utils.set_level(level=level, model="model", spec={"cfgrib": spec})
    assert spec.get("level") == expected


def test_set_level_nlevel_no_level_info_model():
    spec: dict = {"model": {}}
    utils.set_level(level="250mb", model="model", spec={"cfgrib": spec})
    assert spec["model"]["level"] == 250


def test_set_level_nonlevel():
    spec = {"typeOfLevel": "foo"}
    utils.set_level(level="max", model="model", spec={"cfgrib": spec})
    assert spec.get("level") is None


def test_timer_returns_original_value(capsys):
    @utils.timer
    def add(a, b):
        return a + b

    result = add(2, 3)
    captured = capsys.readouterr()
    assert result == 5
    assert "add Elapsed time:" in captured.out
    assert "seconds" in captured.out


def test_timer_preserves_function_name_and_docstring():
    @utils.timer
    def foo():
        """Original docstring."""
        return 42

    assert foo.__name__ == "foo"
    assert foo.__doc__ == "Original docstring."


def test_timer_measures_expected_elapsed_time(capsys):
    @utils.timer
    def slow_func():
        time.sleep(0.01)
        return "done"

    result = slow_func()
    captured = capsys.readouterr()
    assert result == "done"
    # It should print something like: "slow_func Elapsed time: 0.0101 seconds"
    assert "slow_func Elapsed time:" in captured.out


def test_timer_with_mocked_perf_counter(capsys):
    """Make timing deterministic using mocks."""
    with patch("time.perf_counter", side_effect=[10.0, 12.5]):

        @utils.timer
        def example():
            return "ok"

        result = example()
        captured = capsys.readouterr()

    assert result == "ok"
    assert "example Elapsed time: 2.5000 seconds" in captured.out


def test_to_datetime():
    assert utils.to_datetime("2025103112") == datetime(2025, 10, 31, 12, 0, 0)


def test_uniq_wgrib2_list():
    wgrib2_list_path = Path(__name__).parent.resolve() / "tests" / "data" / "wgrib2_submsg1.txt"
    fields_list = wgrib2_list_path.read_text().split("\n")
    uniq_list = utils.uniq_wgrib2_list(fields_list)
    assert len(uniq_list) < len(fields_list)
    assert len(uniq_list) == 1711

