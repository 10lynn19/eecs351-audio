import pathlib, soundata
root=pathlib.Path("data/raw/urbansound8k").resolve()
ds=soundata.initialize('urbansound8k',data_home=str(root))
ds.download()
print(ds.validate())