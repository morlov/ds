import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm
from IPython.display import Markdown
import matplotlib.pyplot as plt

class Binner(BaseEstimator, TransformerMixin):
    
    def __init__(self, nbins=20):
        self.nbins = nbins

    def fit(self, df, y=None):
        self.names = df.columns
        self.feature_bins = {}
        for f in tqdm.tqdm(df.columns):
            nbins = self.nbins
            nvals = len(df[f].unique())
            nbins = min(nbins,  nvals)
            while nbins > 0:
                try:
                    pd.qcut(df[f], q=nbins)
                    break
                except Exception:
                    nbins -= 1
                    
            if nbins == 0:
                warnings.warn("Number of bins set to 0 for " + f)
                self.feature_bins[f] = [np.nan]
                
            _, bins = pd.qcut(df[f], q=nbins, retbins=True, labels=range(nbins))
            self.feature_bins[f] = bins
        return self
            

    def transform(self, df):
        df_ret = df.copy()
        for f, bins in tqdm.tqdm(self.feature_bins.iteritems()):
            labels = range(len(bins)-1)
            if len(bins) > 0:
                df_ret[f] = pd.to_numeric(pd.cut(df[f], bins=bins, include_lowest=True, labels=labels))
        return df_ret
    
    def get_feature_names(self):
        return self.names.tolist()

def psi(pk, qk):
    if type(pk) != type(qk):
        return np.nan
    if len(pk) != len(qk):
        raise ValueError('len(pk) != len(qk)')
    pk, qk = np.array(pk), np.array(qk)
    m = (pk == 0) & (qk == 0)
    pk, qk = pk[~m], qk[~m]
    if any((pk == 0) | (qk == 0)):
        pk = (np.array(pk) + 1.0) / (1 + len(pk))
        qk = (np.array(qk) + 1.0) / (1 + len(qk))
    if sum(pk) < 1 - 1.e-2:
        raise ValueError('sum(pk) = {} != 1.'.format(sum(pk)))
    if sum(qk) < 1 - 1.e-2:
        raise ValueError('sum(qk) = {} != 1.'.format(sum(qk)))
    return np.sum((pk - qk) * np.log(pk / qk), axis=0) 


def binns_report(df_binns, f, time_axis, title=None):
    dfg = df_binns.fillna(-1).groupby([time_axis, f]).agg({"id": np.size}).reset_index()
    dfg.sort_values(by=time_axis, inplace=True)
     
    for t in dfg[time_axis]:
        ix = dfg[time_axis]==t
        dfg.loc[ix, "share"] = dfg.loc[ix, "id"]/sum(dfg.loc[ix, "id"])
    
    df_pivot = dfg.pivot(time_axis, f)['share']
    df_pivot = df_pivot.reset_index().fillna(0)
    df_pivot['values'] = df_pivot.iloc[:, 1:].values.tolist()
    df_pivot['psi'] = [psi(x[0], x[1]) for x in zip(df_pivot.shift(1)['values'], df_pivot['values'])]
    max_psi = max(df_pivot['psi'].fillna(-1))
    
    if max_psi>0.25:
        display(Markdown('# PSI for ' + f +'  = ' + str(max_psi)))
        ax = df_pivot.plot(kind='bar', stacked=True)
        df_pivot.plot(x=time_axis, y="psi", ax=ax, color="black")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(ax.get_xticks(), rotation=90)
        xmax, xmin = 0, df_pivot.shape[0]
        plt.plot([xmin, xmax], [0.1, 0.1], 'b', ls="--")
        plt.plot([xmin, xmax], [0.25, 0.25], 'r', ls="--")
        plt.gca().set_ylim([0, 1])
        if title:
            plt.title(title + " psi = " + str(max_psi))
        plt.show()
        return f
