import pandas as pd
import numpy as np

def entropy3(labels, base=np.e):
    # Reference: https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc)/np.log(base)).sum()