"""
Debug helpers.
"""
import io
import logging
from typing import Union, Optional, Callable

import numpy as np
import pandas as pd


_printt_log_method = print


def set_log_method(log_method: Optional[Callable] = None) -> None:
    global _printt_log_method  # pylint: disable=global-statement
    if log_method is not None:
        _printt_log_method = log_method
    else:
        _printt_log_method = print


def _init_str(minimal: bool = False) -> str:
    if minimal:
        return ""
    global _printt_log_method  # pylint: disable=global-statement
    return "\n" if _printt_log_method == print else "\n\n"  # pylint: disable=comparison-with-callable


force_minimal_logging = False


def ar(
    array: np.ndarray,
    name: Optional[str] = None,
    lim: Union[int, str, None] = None,
    lw: int = 200,
    minimal: bool = False,
) -> None:
    """Debug `ar`ray.
    Print helper for `numpy.ndarray`, will print like so:
    ```
    my_array [<class 'numpy.ndarray'>] [dtype=float32]:
    SHAPE: (3, 3)
    [[ 0.5372,  1.2580, -0.9479],
     [-0.7958, -1.6064, -1.2641],
     [ 1.6119,  1.3587, -0.1000]])
    ```
    The `linewith` printoption will be set to `200` by default (`lw` argument) to allow for fewer line breaks.

    Args:
        array (np.ndarray): array to print.
        name (Optional[str], optional): The name for the array to print. Defaults to None.
        lim (Optional[int, str], optional): If `int`, will set `edgeitems` printoption to this value. 
            If set to `"full"` will print the entire array (can be slow). Defaults to None.
        lw (int, optional): Set the `linewith` printoption to this. Defaults to 200.
        minimal (bool, optional): If true, will not print the array itself. Defaults to False.
    """
    global _printt_log_method  # pylint: disable=global-statement

    if force_minimal_logging:
        minimal = True

    if name is None:
        name = f"Array-{id(array)}"

    content = _init_str(minimal)

    if not minimal:
        content += f"=== <{name}> ===:\n[{type(array)}] [dtype={array.dtype}]\n"
        content += f"SHAPE: {tuple(array.shape)}\n"

        with np.printoptions(
            threshold=np.product(array.shape) if lim == "full" else 1000,  # 1000 is default.
            edgeitems=lim if isinstance(lim, int) else 3,  # 3 is default.
            linewidth=lw,
        ):
            content += str(array)
        content += "\n"  # Leave one blank line after printing.

    else:
        content += f"<{name}>:: {array.shape}"

    _printt_log_method(content)


def ar_(*args, **kwargs):
    """
    Shortcut for `ar(..., minimal=True)`.
    """
    ar(*args, **kwargs, minimal=True)


def setup_logger(
    name: str, level: int = logging.INFO, format_str: str = "%(name)s:%(levelname)s:\t%(message)s"
) -> logging.Logger:
    """Set up a console logger with name `name`.

    Args:
        name (str): Logger name.
        level (int): Logging level to set. Defaults to logging.INFO.
        format_str (str, optional): The format string to use for the logger formatter. 
            Defaults to "%(name)s:%(levelname)s:\t%(message)s".

    Returns:
        logging.Logger: [description]
    """
    _logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt=format_str)
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(level)
    return _logger


def _df_info_to_str(dataframe: pd.DataFrame) -> str:
    buf = io.StringIO()
    dataframe.info(buf=buf)
    return buf.getvalue()


def df(
    dataframe: Union[pd.DataFrame, pd.Series],
    name: Optional[str] = None,
    info: bool = False,
    max_rows_before_collapse: Optional[Union[int, str]] = None,
    keep_rows_if_collapsed: Optional[int] = None,
    force_show_all_cols: bool = False,
    minimal: bool = False,
) -> None:
    """Debug `d`ata`f`rame. 
    Print helper for `pd.DataFrame`.
    """
    global _printt_log_method  # pylint: disable=global-statement

    if force_minimal_logging:
        minimal = True

    if name is None:
        name = f"DataFrame-{id(dataframe)}"

    if isinstance(dataframe, pd.DataFrame):
        tp = "<class 'pd.DataFrame'>"
    elif isinstance(dataframe, pd.Series):
        tp = "<class 'pd.Series'>"
    else:
        raise ValueError(f"`df` must be a pandas DataFrame or Series, was {type(dataframe)}.")

    content = _init_str(minimal)

    if not minimal:
        content += f"=== <{name}> ===:\n[{tp}]\n\n"
        pd_option_seq = []
        if max_rows_before_collapse is not None:
            if max_rows_before_collapse == "full":
                max_rows_before_collapse = dataframe.shape[0]
            pd_option_seq.extend(["display.max_rows", max_rows_before_collapse])
        if keep_rows_if_collapsed is not None:
            pd_option_seq.extend(["display.min_rows", keep_rows_if_collapsed])
        if force_show_all_cols:
            pd_option_seq.extend(["display.max_columns", dataframe.shape[1]])
            pd_option_seq.extend(["display.expand_frame_repr", True])

        def _build(c):
            if info:
                c += _df_info_to_str(dataframe) + "\n"
            c += str(dataframe) + "\n"
            return c

        if len(pd_option_seq) > 0:
            with pd.option_context(*pd_option_seq):
                content = _build(content)
        else:
            content = _build(content)

    else:
        content += f"<{name}>:: {tp}:: {dataframe.shape}"

    _printt_log_method(content)


def df_(*args, **kwargs):
    """
    Shortcut for `df(..., minimal=True)`.
    """
    df(*args, **kwargs, minimal=True)
