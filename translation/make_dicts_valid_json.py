import re

with open('./models/dicts/input_iw.json', 'r') as f, open('./models/dicts/test-input_iw.json', 'x') as o:
  print(re.sub(r'[0-9]+', r'"\g<0>"', re.sub(r"'", r'"', f.read())), file=o)
with open('./models/dicts/target_iw.json', 'r') as f, open('./models/dicts/test-target_iw.json', 'x') as o:
  print(re.sub(r'[0-9]+', r'"\g<0>"', re.sub(r"'", r'"', f.read())), file=o)