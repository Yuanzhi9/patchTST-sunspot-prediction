Solar Cycle Optimization Framework Summary
======================================================================
Detected cycles:
 cycle start_date end_date  complete  length_months peak_date  peak_month_in_cycle  peak_ssn_smooth  start_ssn_smooth
    12    1878-10  1888-11      True            122   1883-12                   62       123.015385          4.023077
    13    1888-12  1901-12      True            157   1894-01                   61       147.746154          9.007692
    14    1902-01  1913-06      True            138   1906-01                   48       108.907692          4.238462
    15    1913-07  1923-06      True            120   1917-07                   48       172.161538          2.592308
    16    1923-07  1933-08      True            122   1928-03                   56       129.100000          8.976923
    17    1933-09  1944-02      True            126   1937-04                   43       197.069231          6.269231
    18    1944-03  1954-03      True            121   1947-05                   38       216.815385         13.546154
    19    1954-04  1964-09      True            126   1958-03                   47       286.892308          5.592308
    20    1964-10  1976-05      True            140   1968-11                   49       158.038462         14.046154
    21    1976-06  1986-02      True            117   1979-12                   42       231.707692         17.823077
    22    1986-03  1996-07      True            125   1989-06                   39       212.769231         13.592308
    23    1996-08  2008-10      True            147   2002-03                   67       180.915385         11.053846
    24    2008-11  2019-11      True            133   2014-04                   65       115.330769          2.376923
    25    2019-12  partial     False             71   2024-10                   58       159.223077          1.938462

Cycle 24 / Cycle 25 validation metrics sorted by MAE:
 cycle                               model  obs_months_used       MAE      RMSE         R         R2       Bias       MAPE   N
    24 M4 parametric Waldmeier curve (60m)               60 10.891410 14.179964  0.993839   0.863031  -4.864035  53.981005  73
    24 M4 parametric Waldmeier curve (48m)               48 20.196189 27.344476  0.896712   0.515384 -19.917347  37.404514  85
    24 M4 parametric Waldmeier curve (36m)               36 32.854717 34.426928  0.966787   0.220599  32.854717 171.928816  97
    24     M3 calibrated early shape (48m)               48 40.376955 46.441880  0.682577  -0.397906  35.596075 355.256944  85
    24         M2 early shape analog (60m)               60 41.099390 45.565926  0.850475  -0.414340  39.249219 377.003804  73
    24     M3 calibrated early shape (60m)               60 46.139318 50.197795  0.674465  -0.716496  40.758084 417.912402  73
    24     M3 calibrated early shape (36m)               36 46.835699 53.426796  0.704717  -0.877078  45.567027 360.610158  97
    24         M2 early shape analog (48m)               48 50.013380 55.404082  0.826904  -0.989491  50.013380 404.289160  85
    24           M1 prior successor analog                0 58.440464 69.748745  0.699247  -2.367933  58.423515 375.531521 133
    24         M2 early shape analog (36m)               36 73.691646 78.297774  0.737793  -3.031466  73.691646 478.852177  97
    25 M4 parametric Waldmeier curve (36m)               36  6.144367  7.343676  0.850456   0.710396   0.030566   4.508830  35
    25 M4 parametric Waldmeier curve (48m)               48  7.923253  9.448665  0.952574   0.343055  -3.752606   5.492546  23
    25           M1 prior successor analog                0 10.370518 14.372121  0.968236   0.928233  -0.811100  31.384629  71
    25 M4 parametric Waldmeier curve (60m)               60 15.475528 17.371048  0.914660  -2.216732  15.475528  12.113976  11
    25     M3 calibrated early shape (60m)               60 16.710985 19.171596 -0.982946  -2.918134  10.818580  13.005496  11
    25         M2 early shape analog (48m)               48 18.534885 25.029856 -0.310560  -3.610044  18.534885  14.175439  23
    25     M3 calibrated early shape (36m)               36 22.001313 24.099526  0.414147  -2.118849 -19.073004  16.223292  35
    25     M3 calibrated early shape (48m)               48 23.482044 26.558184 -0.614998  -4.190211 -19.178731  16.214072  23
    25         M2 early shape analog (60m)               60 27.874901 30.539440 -0.963188  -8.942255  27.874901  21.732406  11
    25         M2 early shape analog (36m)               36 41.950608 46.587970  0.676394 -10.655338  41.950608  31.253492  35

Peak metrics sorted by peak amplitude absolute error:
 cycle                               model  obs_months_used true_peak_date pred_peak_date  peak_month_error  true_peak_ssn  pred_peak_ssn  peak_amp_error  peak_amp_abs_error
    24     M3 calibrated early shape (48m)               48        2014-04        2015-05                13     115.330769     114.609272       -0.721498            0.721498
    24     M3 calibrated early shape (60m)               60        2014-04        2015-10                18     115.330769     109.731226       -5.599543            5.599543
    24         M2 early shape analog (48m)               48        2014-04        2014-10                 6     115.330769     124.164551        8.833782            8.833782
    24         M2 early shape analog (60m)               60        2014-04        2014-10                 6     115.330769     105.292322      -10.038447           10.038447
    24     M3 calibrated early shape (36m)               36        2014-04        2014-11                 7     115.330769     138.121511       22.790742           22.790742
    24 M4 parametric Waldmeier curve (48m)               48        2014-04        2012-11               -17     115.330769      90.777371      -24.553398           24.553398
    24 M4 parametric Waldmeier curve (36m)               36        2014-04        2013-09                -7     115.330769     140.293555       24.962785           24.962785
    24 M4 parametric Waldmeier curve (60m)               60        2014-04        2013-11                -5     115.330769      89.741480      -25.589289           25.589289
    24         M2 early shape analog (36m)               36        2014-04        2014-03                -1     115.330769     165.981231       50.650462           50.650462
    24           M1 prior successor analog                0        2014-04        2014-04                 0     115.330769     172.635463       57.304694           57.304694
    25 M4 parametric Waldmeier curve (60m)               60        2024-12        2024-12                 0     150.300000     150.688191        0.388191            0.388191
    25           M1 prior successor analog                0        2024-10        2025-05                 7     159.223077     160.131392        0.908315            0.908315
    25     M3 calibrated early shape (60m)               60        2024-12        2025-10                10     150.300000     152.995502        2.695502            2.695502
    25         M2 early shape analog (48m)               48        2024-10        2025-10                12     159.223077     169.109834        9.886757            9.886757
    25         M2 early shape analog (60m)               60        2024-12        2025-10                10     150.300000     163.600224       13.300224           13.300224
    25 M4 parametric Waldmeier curve (36m)               36        2024-10        2024-08                -2     159.223077     143.367206      -15.855871           15.855871
    25 M4 parametric Waldmeier curve (48m)               48        2024-10        2024-09                -1     159.223077     140.100326      -19.122751           19.122751
    25     M3 calibrated early shape (48m)               48        2024-10        2025-10                12     159.223077     137.216036      -22.007041           22.007041
    25     M3 calibrated early shape (36m)               36        2024-10        2025-10                12     159.223077     137.013671      -22.209406           22.209406
    25         M2 early shape analog (36m)               36        2024-10        2025-04                 6     159.223077     201.062877       41.839800           41.839800

Best method by cycle:
 cycle                               model  obs_months_used       MAE      RMSE        R       R2      Bias
    24 M4 parametric Waldmeier curve (60m)               60 10.891410 14.179964 0.993839 0.863031 -4.864035
    25 M4 parametric Waldmeier curve (36m)               36  6.144367  7.343676 0.850456 0.710396  0.030566

Cycle 26 scenario summary:
                                 item   value
                        cycle25_start 2019-12
                   latest_observation 2025-10
              estimated_cycle26_start 2034-12
   estimated_cycle26_medium_peak_date 2040-05
                cycle26_weak_peak_ssn  114.70
              cycle26_medium_peak_ssn  167.91
              cycle26_strong_peak_ssn  215.35
   cycle26_weighted_median_peak_month      48
cycle26_weighted_median_length_months     126
      cycle25_waldmeier_pred_peak_amp  159.22
    cycle25_waldmeier_pred_peak_month    58.0
        cycle25_waldmeier_pred_length   135.5

Interpretation:
1. M1 is the strict prior forecast: it tests whether one can forecast a whole next cycle using only the previous cycle. This is intentionally hard and often underestimates strong cycles.
2. M2 uses early observations of the target cycle, therefore it is a current-cycle update method. It should be judged mainly on future_only metrics after the observation window.
3. M3 adds Waldmeier-effect style calibration, using early rise speed and early amplitude to correct peak amplitude, peak timing and cycle length. Under single-variable SSN constraints, this is the main optimization route.
4. M4 uses a parametric cycle curve with Waldmeier priors and is especially useful for long decline forecasting.
5. Cycle 24 and Cycle 25 must be compared together. If a method performs well on Cycle 24 but underestimates Cycle 25 peak, the error is likely related to strong-cycle amplitude calibration rather than ordinary curve fitting.
6. Cycle 26 should be reported as weak/medium/strong scenarios unless additional precursor data such as polar field, F10.7, sunspot area, aa/Ap is included.