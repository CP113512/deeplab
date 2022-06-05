from pathlib import Path

root = Path('./datasets-cp/JPEGImages')
pathlist = [path for path in root.iterdir()]
for path in pathlist:
    if path.suffix == '.JPG':
        new_name = str(path).split('.')[0] + '.jpg'
        path.rename(new_name)