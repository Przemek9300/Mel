
import pandas as pd

df = pd.read_csv('clean.csv')

# print(df.mask(lambda x: x[0]))
CLASSES_LABEL = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

for header in ['sex', 'age', 'localization']:
    for label in CLASSES_LABEL:
        print(label)
        df[(df.dx == label)].groupby([header]).size().to_json(
            f'./chart/{label}-{header}.json')
# print(df)
