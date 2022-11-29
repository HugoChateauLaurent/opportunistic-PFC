def NEURON_COUNTS():
    N = {}
    N["ECin"]   = 110000
    N["DG"]     = 1200000
    N["CA3"]    = 250000
    N["CA1"]    = 390000
    N["ECout"]  = 330000
    return N

READABLE_TARGET_LABEL = {
    None: "None",
    tuple(["ECin"]): "ECin",
    tuple(["DG"]): "DG",
    tuple(["CA3"]): "CA3",
    tuple(["CA1"]): "CA1",
    tuple(["ECout"]): "ECout",
    tuple(["ECin","CA1"]): "ECin+CA1",
    tuple(["CA1","ECout"]): "CA1+ECout",
    tuple(["ECin","CA1","ECout"]): "Bio+", # Hypothesized biological configuration
    tuple(["ECin","DG","CA1","ECout"]): "Bio", # Hypothesized biological configuration
    tuple(["ECin","DG","CA3","CA1","ECout"]): "All"
}

BIO_INDEX = list(READABLE_TARGET_LABEL.values()).index("Bio")

MODULAR_LAYERS = ["ECin", "DG", "CA3", "CA1", "ECout"]



READABLE_MECHANISMS = {
    None: "None",
    0: "Weight avg",
    1: "Weight sum",
    2: "Bias avg",
    3: "Bias sum",
    4: "Activity avg",
    5: "Activity sum",
    6: "Input"
}
