import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conv_zoo(filename = "conv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    conv_df = pd.read_csv(filename)
    hws = conv_df["input_h"]
    cins = conv_df["cin"]
    couts = conv_df["cout"]
    ks = conv_df["ks"]
    strides = conv_df["stride"]
    return hws, cins, couts, ks, strides


def read_conv_nn_filtering_zoo(filename="conv_nn_filtering.csv"):
    """Standard-conv configs from NN_Filtering LOP/VLOP/HOP families (144x144 patch models)."""
    return read_conv_zoo(filename)


def read_dwconv_zoo(filename = "dwconv.csv"):
    filename = os.path.join(BASE_DIR, filename)
    dwconv_df = pd.read_csv(filename)
    if "kh" in dwconv_df.columns:
        dwconv_df = dwconv_df.loc[dwconv_df["kh"].isna()]
    hws = dwconv_df["input_h"]
    cins = dwconv_df["cin"]
    ks = dwconv_df["ks"]
    strides = dwconv_df["stride"]
    return hws, cins, ks, strides


def read_separable_dwconv_zoo(filename="dwconv.csv"):
    """Separable depthwise legs from dwconv.csv (rows with kh,kw,sh,sw columns set)."""
    filename = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(filename)
    if "kh" not in df.columns:
        return [], [], [], [], [], [], [], []
    sub = df.loc[df["kh"].notna()].copy()
    input_hs = sub["input_h"].astype(int).tolist()
    input_ws = sub["input_w"].astype(int).tolist()
    cins = sub["cin"].astype(int).tolist()
    khs = sub["kh"].astype(int).tolist()
    kws = sub["kw"].astype(int).tolist()
    shs = sub["sh"].astype(int).tolist()
    sws = sub["sw"].astype(int).tolist()
    groups = sub["groups"].astype(int).tolist()
    return input_hs, input_ws, cins, khs, kws, shs, sws, groups


def read_separable_dwconv_prior_configs(filename="dwconv.csv"):
    """Unique separable-dwconv configs from NN_Filtering rows in dwconv.csv."""
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)
    if "kh" not in df.columns:
        return [], [], [], [], [], [], [], []
    sub = df.loc[df["kh"].notna()]

    configs = []
    seen = set()
    input_hs, input_ws, cins = set(), set(), set()
    khs, kws, shs, sws = set(), set(), set(), set()
    for _, row in sub.iterrows():
        ih = int(row["input_h"])
        iw = int(row["input_w"])
        cin = int(row["cin"])
        kh = int(row["kh"])
        kw = int(row["kw"])
        sh = int(row["sh"])
        sw = int(row["sw"])
        input_hs.add(ih)
        input_ws.add(iw)
        cins.add(cin)
        khs.add(kh)
        kws.add(kw)
        shs.add(sh)
        sws.add(sw)
        key = (ih, iw, cin, kh, kw, sh, sw)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "INPUT_H": ih,
            "INPUT_W": iw,
            "HW": max(ih, iw),
            "CIN": cin,
            "KERNEL_H": kh,
            "KERNEL_W": kw,
            "STRIDE_H": sh,
            "STRIDE_W": sw,
        })
    return (
        configs,
        sorted(input_hs),
        sorted(input_ws),
        sorted(cins),
        sorted(khs),
        sorted(kws),
        sorted(shs),
        sorted(sws),
    )


def read_dwconv_nn_filtering_zoo(filename="dwconv_nn_filtering.csv"):
    """Depthwise-conv configs from NN_Filtering LOP/VLOP families (144x144 patch models)."""
    return read_dwconv_zoo(filename)


def read_nn_filtering_dwconv_prior_configs(filename="dwconv.csv"):
    """Unique dwconv configs and allowlists from LOP/VLOP/HOP rows in dwconv.csv.

    Returns:
        configs: list of dicts with HW, CIN, KERNEL_SIZE, STRIDES (deduplicated)
        hws, cins, kernel_sizes, strides: sorted unique dimension values for data_validation
    """
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)
    mask = df["model"].astype(str).str.match(r"^(LOP|VLOP|HOP)", na=False)
    sub = df.loc[mask]

    configs = []
    seen = set()
    hws, cins, kernel_sizes, strides = set(), set(), set(), set()
    for _, row in sub.iterrows():
        hw = int(row["input_h"])
        cin = int(row["cin"])
        ks = int(row["ks"])
        stride = int(row["stride"])
        hws.add(hw)
        cins.add(cin)
        kernel_sizes.add(ks)
        strides.add(stride)
        key = (hw, cin, ks, stride)
        if key in seen:
            continue
        seen.add(key)
        configs.append({
            "HW": hw,
            "CIN": cin,
            "KERNEL_SIZE": ks,
            "STRIDES": stride,
        })
    return (
        configs,
        sorted(hws),
        sorted(cins),
        sorted(kernel_sizes),
        sorted(strides),
    )


def read_fc_zoo(filename = "fc.csv"):
    filename = os.path.join(BASE_DIR, filename)
    fc_df = pd.read_csv(filename)
    cins = fc_df["cin"]
    couts = fc_df["cout"]
    return cins, couts


def read_pool_zoo(filename = "pooling.csv"):
    filename = os.path.join(BASE_DIR, filename)
    pool_df = pd.read_csv(filename)
    hws = pool_df["input_h"]
    cins = pool_df["cin"]
    ks = pool_df["ks"]
    strides = pool_df["stride"]
    return hws, cins, ks, strides
