import pandas as pd
import numpy as np

import common.data.data_preprocess as prp


DATA = [
    [1,  200, np.nan, 1.1,    33.1], 
    [1,  300, np.nan, 2.2,    44.1], 
    [1,  400, 10.5,   3.3,    55.1], 
    [88, 333, 22.1,   np.nan, 40.3], 
    [88, 433, np.nan, 7.7,    50.4], 
    [88, 533, np.nan, 8.8,    60.5], 
    [88, 633, -22.3,  np.nan, np.nan], 
    [88, 733, 112.4,  np.nan, 70.6], 
    [7,  599, np.nan,   4.3,    22.3], 
    [7,  699, np.nan, np.nan, np.nan]
]
LOADED_DATA_EXPECTED = [
    [
        [ -1., -1.,     -1.,    -1.   ],
        [200., np.nan,  1.1,    33.1  ],
        [300., np.nan,  2.2,    44.1  ],
        [400., 10.5,    3.3,    55.1  ],
    ],
    [
        [ -1., -1.,     -1.,    -1.   ],
        [ -1., -1.,     -1.,    -1.   ],
        [599., np.nan,    4.3,    22.3  ],
        [699., np.nan,  np.nan, np.nan],
    ],
    [
        [333., 22.1,    np.nan, 40.3  ],
        [433., np.nan,  7.7,    50.4  ],
        [533., np.nan,  8.8,    60.5  ],
        [633., -22.3,   np.nan, np.nan],
    ],
]
PADDING_MASK_EXPECTED = [
    [
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ],
    [
        [ True,  True,  True,  True],
        [ True,  True,  True,  True],
        [False, False, False, False],
        [False, False, False, False]
    ],
    [
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]
]
MEDIAN_VALS_EXPECTED = [433., 10.5, 3.8, 44.1]
SCALER_DATA_MIN_EXPECTED = [200., -22.3, 1.1, 22.3]
SCALER_DATA_RANGE_EXPECTED = [499., 44.4, 7.7, 38.2]
PREPROC_DATA_EXPECTED = [
    [
        [-1.,   -1.,            -1.,            -1.           ],  # Padding (-1)
        [ 0.,   np.nan,         0.,             2.82722513e-01],
        [ 100., np.nan,         1.42857143e-01, 5.70680628e-01],
        [ 200., 7.38738739e-01, 2.85714286e-01, 8.58638743e-01],
    ],
    [
        [-1.,   -1.,            -1.,            -1.           ],  # Padding (-1)
        [-1.,   -1.,            -1.,            -1.           ],  # Padding (-1)
        [ 0.,   np.nan,         4.15584416e-01, 0.            ],
        [ 100., np.nan,         np.nan,         np.nan        ],
    ],
    [
        [ 0.,   1.,             np.nan,         4.71204188e-01],
        [ 100., np.nan,         8.57142857e-01, 7.35602094e-01],
        [ 200., np.nan,         1.,             1.            ],
        [ 300., 0.,             np.nan,         np.nan        ],
    ]
]
IMPUTED_DATA_EXPECTED = [
    [
        [-1.,   -1.,            -1.,            -1.           ],
        [ 0.,   7.38738739e-01, 0.,             2.82722513e-01],  # bfill (7.38738739e-01)
        [ 100., 7.38738739e-01, 1.42857143e-01, 5.70680628e-01],  # bfill (7.38738739e-01)
        [ 200., 7.38738739e-01, 2.85714286e-01, 8.58638743e-01],
    ],
    [
        [-1.,   -1.,            -1.,            -1.           ],
        [-1.,   -1.,            -1.,            -1.           ],
        [ 0.,   7.38738739e-01,  4.15584416e-01, 0.           ],  
        # ^ median fill: scaled(10.5)=7.38738739e-01
        [ 100., 7.38738739e-01,  4.15584416e-01, 0.           ],  
        # ^ median fill: scaled(10.5)=7.38738739e-01, ffill (4.15584416e-01, 0.)
    ],
    [
        [ 0.,   1.,             8.57142857e-01, 4.71204188e-01],  # bfill (8.57142857e-01)
        [ 100., 0.,             8.57142857e-01, 7.35602094e-01],  # bfill (0.)
        [ 200., 0.,             1.,             1.            ],  # bfill (0.)
        [ 300., 0.,             1.,             1.            ],  # ffill (1.)
    ]
]


def _prep_file(filepath):
    df = pd.DataFrame(data=DATA, columns=["admissionid", "time", "6001", "6002", "6003"])
    df.to_csv(filepath, index=False)


def test_load_and_reshape(tmp_path):
    """Test the load and reshape process.

    Args:
        tmp_path (pathlib2.Path): temporary testing directory.
    """
    filepath = tmp_path / "df.csv"
    _prep_file(filepath)
    loaded_data, padding_mask = prp.load_and_reshape(filepath, max_seq_len=4)

    np.testing.assert_array_almost_equal(loaded_data, LOADED_DATA_EXPECTED)
    np.testing.assert_array_almost_equal(padding_mask, PADDING_MASK_EXPECTED)


def test_preprocess_data(tmp_path):
    """Test the main preprocessing function.

    Args:
        tmp_path (pathlib2.Path): temporary testing directory.
    """
    filepath = tmp_path / "df.csv"
    _prep_file(filepath)
    data, padding_mask = prp.load_and_reshape(filepath, max_seq_len=4)

    median_vals = prp.get_medians(data, padding_mask)
    imputed_data = prp.impute(data, padding_mask, median_vals)
    scaler_i = prp.get_scaler(imputed_data, padding_mask)
    imputed_processed_data = prp.process(imputed_data, padding_mask, scaler_i)
    scaler_o = prp.get_scaler(data, padding_mask)
    processed_data = prp.process(data, padding_mask, scaler_o)

    np.testing.assert_array_almost_equal(median_vals, MEDIAN_VALS_EXPECTED)
    np.testing.assert_array_almost_equal(scaler_i.data_min_, SCALER_DATA_MIN_EXPECTED)
    np.testing.assert_array_almost_equal(scaler_i.data_range_, SCALER_DATA_RANGE_EXPECTED)

    np.testing.assert_array_almost_equal(processed_data, PREPROC_DATA_EXPECTED)
    np.testing.assert_array_almost_equal(imputed_processed_data, IMPUTED_DATA_EXPECTED)


def test_preprocess_twice(tmp_path):
    """Test the main preprocessing function, effect of doing it twice.

    Args:
        tmp_path (pathlib2.Path): temporary testing directory.
    """
    filepath = tmp_path / "df.csv"
    _prep_file(filepath)
    data, padding_mask = prp.load_and_reshape(filepath, max_seq_len=4)

    # Once.
    preproc_data, imputed_data = prp.preprocess_data(data, padding_mask)

    # Twice.
    preproc_data_2, imputed_data_2 = prp.preprocess_data(data, padding_mask)

    print(imputed_data)
    print("\n")
    print(imputed_data_2)

    np.testing.assert_array_almost_equal(preproc_data, preproc_data_2)
    np.testing.assert_array_almost_equal(imputed_data, imputed_data_2)
