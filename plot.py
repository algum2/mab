#-*- coding: utf-8 -*-
#File: plot.py
#Author: yobobobo(zhouboacmer@qq.com)
import sys
from collections import defaultdict

res = defaultdict(list)

for fn in sys.argv[1:]:
  with open(fn, 'r') as f:
    rwd = res[fn.strip()]
    for line in f:
        items = line.strip().split('\t')
        rwd.append(items[8].split('=')[1])

keys = sorted(res.keys())
for k in keys:
  print k,
print()

for idx in range(100):
  for k in keys:
    v = res[k]
    if idx < len(v):
      #print(v[idx], end='\t')
      print v[idx],
    else:
      #print('', end='\t')
      print '',
  print()


