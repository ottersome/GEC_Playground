import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    df = pd.read_csv('../wi+locness/classified.csv')

    
    hist = {keys:0 for i,keys in enumerate(df.head()) if i >= 2}
    progress_bar = tqdm(df.iterrows())
    for i, row in progress_bar:
        keys = row.keys()
        for key in keys[2:]:
            hist[key] += row[key]

    hist = dict(sorted(hist.items(),key=lambda item: item[1],reverse=True))
    plt.bar(np.arange(len(hist.keys())), hist.values())
    tick_property = plt.xticks(np.arange(len(hist.keys())),hist.keys(), rotation=45, fontsize=10)
    plt.show()


