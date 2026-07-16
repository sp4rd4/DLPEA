# CERT Insider Threat Test Dataset r4.2

Experiment E2 (`src/run_e2_cert.py`) uses release **r4.2** of the CERT Insider Threat Test Dataset:

> Lindauer B. Insider Threat Test Dataset. Carnegie Mellon University, 2020.
> DOI: [10.1184/R1/12841247.v1](https://doi.org/10.1184/R1/12841247.v1)

The dataset is distributed by Carnegie Mellon University (KiltHub) and is not included in this repository because of its size (the r4.2 archive is about 4.5 GB).

## Required files

Download `r4.2.tar.bz2` and the answers archive from the dataset page, extract them, and place the files as follows:

```text
data/cert/r4.2/logon.csv
data/cert/r4.2/device.csv
data/cert/r4.2/file.csv
data/cert/r4.2/email.csv
data/cert/answers_dir/answers/insiders.csv
```

Notes:

- only the four log files listed above are read; `http.csv` (about 11 GB) is not used and does not need to be extracted;
- `insiders.csv` from the answers archive provides the ground-truth insider scenarios (the rows with `dataset = 4.2`) used to label malicious user-days;
- `src/cert_features.py` also checks `data/cert/need/` as an alternative location for the four log files.
