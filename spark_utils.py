from itertools import chain, islice


def chunks(iterable, size=10000):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def score(columns, model):
    b = dill.loads(model.value)
    def f(it):    
        for chunk in chunks(it):
            df = pd.DataFrame(list(chunk), columns=columns)
            df["proba"] = b.predict_proba(df)[:, 1]
            for z in df[["id", "proba"]].iterrows():
                yield list(z[1])
    return f
