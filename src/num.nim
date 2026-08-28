## num — n-dimensional numeric arrays for Nim, in the spirit of numpy.
##
## ```nim
## import num
##
## let a = arange(12).reshape(3, 4).astype(float)
## echo a[1, All]                 # a row, as a view
## echo a.mean(axis = 0)          # column means
## echo (a > 5.0).sum(axis = 1)   # per-row counts
## echo matmul(a, a.t)
## ```
##
## This is the whole public surface: everything a user should reach is
## re-exported here. The bridges to other libraries are **not** — importing
## `num/racoon` or `num/plotter` is what pulls those packages in, since Nim
## has no optional dependencies.

import num/core
import num/shape
import num/creation
import num/index
import num/slicing
import num/ops
import num/reductions
import num/linalg
import num/stats
import num/random
import num/summary

export core, shape, creation, index, slicing, ops, reductions, linalg,
       stats, random, summary
