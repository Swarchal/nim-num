# num

N-dimensional numeric arrays for Nim, in the spirit of numpy — shapes,
strides, broadcasting, views, elementwise ufuncs, axis reductions, slicing,
linear algebra and statistics. No dependencies.

It is built to sit alongside two other Nim libraries, with an optional bridge
to each:

- **[racoon](https://github.com/Swarchal/Racoon)** — dataframes. Columns come
  across as arrays, arrays go back as columns, a frame becomes a design
  matrix.
- **[plotter](https://github.com/Swarchal/nim-plotter)** — Vega-Lite charts.
  Arrays go straight into `scatter`/`line`/`bar`/`hist`/`heatmap`.

Neither is a dependency: importing the bridge module is what pulls one in.

```sh
nimble install https://github.com/Swarchal/nim-num
```

Nim 2.0 or newer. Every snippet below assumes `import num`, and the comment
under each line is what it actually prints.

## A quick tour

```nim
let a = arange(12).reshape(3, 4).astype(float)

echo a
# array([[ 0.0,  1.0,  2.0,  3.0],
#        [ 4.0,  5.0,  6.0,  7.0],
#        [ 8.0,  9.0, 10.0, 11.0]])

echo a[1, All]                              # a row, as a view
# array([4.0, 5.0, 6.0, 7.0])

echo a[0..1, 1..^1]                         # a 2x3 window
# array([[1.0, 2.0, 3.0],
#        [5.0, 6.0, 7.0]])

echo a.mean(axis = 0)                       # column means
# array([4.0, 5.0, 6.0, 7.0])

echo a - a.mean(axis = 1, keepDims = true)  # rows centred
# array([[-1.5, -0.5,  0.5,  1.5],
#        [-1.5, -0.5,  0.5,  1.5],
#        [-1.5, -0.5,  0.5,  1.5]])

echo sum(a > 5.0, axis = 1)                 # per-row counts
# array([0, 2, 4])

echo a[a > 8.0]                             # what a mask selects, flattened
# array([ 9.0, 10.0, 11.0])

echo matmul(a, a.t)
# array([[ 14.0,  38.0,  62.0],
#        [ 38.0, 126.0, 214.0],
#        [ 62.0, 214.0, 366.0]])
```

## Creating arrays

The element type is always explicit for the "filled" constructors —
`zeros[float](2, 3)`, not `zeros(2, 3)`. `arange` and `linspace` infer theirs
from their arguments.

```nim
echo toNDArray(@[1.0, 2.0, 3.0])
# array([1.0, 2.0, 3.0])

echo arr(@[@[1, 2, 3], @[4, 5, 6]])        # arr is the short alias
# array([[1, 2, 3],
#        [4, 5, 6]])

echo zeros[float](2, 3)
# array([[0.0, 0.0, 0.0],
#        [0.0, 0.0, 0.0]])

echo ones[int](2, 2)
# array([[1, 1],
#        [1, 1]])

echo full(@[2, 2], 7)
# array([[7, 7],
#        [7, 7]])
```

```nim
echo arange(6)
# array([0, 1, 2, 3, 4, 5])

echo arange(1, 10, 3)
# array([1, 4, 7])

echo arange(1.0, 2.0, 0.25)
# array([ 1.0, 1.25,  1.5, 1.75])

echo linspace(0.0, 1.0, 5)
# array([ 0.0, 0.25,  0.5, 0.75,  1.0])

echo logspace(0.0, 3.0, 4)
# array([   1.0,   10.0,  100.0, 1000.0])

echo eye[float](3)
# array([[1.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0],
#        [0.0, 0.0, 1.0]])

echo eye[int](2, 3, 1)                     # ones on the k-th diagonal
# array([[0, 1, 0],
#        [0, 0, 1]])
```

`meshgrid` gives the two coordinate matrices, as broadcast views — they cost
nothing until something writes to them:

```nim
let (xx, yy) = meshgrid(arange(4), arange(3))

echo xx
# array([[0, 1, 2, 3],
#        [0, 1, 2, 3],
#        [0, 1, 2, 3]])

echo yy
# array([[0, 0, 0, 0],
#        [1, 1, 1, 1],
#        [2, 2, 2, 2]])

echo xx + yy * 10
# array([[ 0,  1,  2,  3],
#        [10, 11, 12, 13],
#        [20, 21, 22, 23]])
```

## Shape, views and copies

An `NDArray[T]` is a `shape`, a `strides` and an `offset` over a shared
buffer. Reshaping, transposing, slicing and broadcasting all produce another
window onto the same buffer, so they are O(1). `describe` shows what a view
actually is:

```nim
let a = arange(12)

echo a.reshape(3, 4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

echo a.reshape(2, -1).shape                # one dimension may be inferred
# @[2, 6]

echo a.reshape(3, 4).t
# array([[ 0,  4,  8],
#        [ 1,  5,  9],
#        [ 2,  6, 10],
#        [ 3,  7, 11]])

echo a.reshape(3, 4).describe
# NDArray[int] shape=(3, 4) strides=(4, 1) contiguous

echo a.reshape(3, 4).t.describe
# NDArray[int] shape=(4, 3) strides=(1, 4) strided

echo a.reshape(3, 4)[All, 1..2].describe
# NDArray[int] shape=(3, 2) strides=(4, 1) offset=1 strided
```

Views are **write-through**. `copy` is the one operation that always
detaches:

```nim
var a = zeros[int](3, 3)
var row = a[1, All]
row += 5                                   # writes into a

echo a
# array([[0, 0, 0],
#        [5, 5, 5],
#        [0, 0, 0]])

a[All, 0] = 9

echo a
# array([[9, 0, 0],
#        [9, 5, 5],
#        [9, 0, 0]])

let detached = a[0, All].copy()
a[0, 0] = -1

echo detached                              # unchanged
# array([9, 0, 0])

echo a
# array([[-1,  0,  0],
#        [ 9,  5,  5],
#        [ 9,  0,  0]])
```

Joining and casting:

```nim
let a = arange(6).reshape(2, 3)

echo concat(@[a, a], 0)
# array([[0, 1, 2],
#        [3, 4, 5],
#        [0, 1, 2],
#        [3, 4, 5]])

echo concat(@[a, a], 1)
# array([[0, 1, 2, 0, 1, 2],
#        [3, 4, 5, 3, 4, 5]])

echo stack(@[arange(3), arange(3) * 10], 0)   # a new axis
# array([[ 0,  1,  2],
#        [ 0, 10, 20]])

echo stack(@[arange(3), arange(3) * 10], 1)
# array([[ 0,  0],
#        [ 1, 10],
#        [ 2, 20]])

echo arange(4).astype(float)
# array([0.0, 1.0, 2.0, 3.0])

echo (arange(4) > 1).astype(int)
# array([0, 0, 1, 1])
```

Also here: `ravel`, `flatten`, `swapAxes`, `moveAxis`, `expandDims`,
`squeeze`, `broadcastTo`, `vstack`, `hstack`.

`flip` reverses an axis. Like `transpose` it is a view — the axis keeps its
length and its stride changes sign — so it costs nothing and it aliases:

```nim
let a = arange(12).reshape(3, 4)

echo flip(a, 0)                           # rows, bottom to top
# array([[ 8,  9, 10, 11],
#        [ 4,  5,  6,  7],
#        [ 0,  1,  2,  3]])

echo flip(a, 1)                           # columns, right to left
# array([[ 3,  2,  1,  0],
#        [ 7,  6,  5,  4],
#        [11, 10,  9,  8]])

echo flip(a)                              # every axis: a 180-degree turn
# array([[11, 10,  9,  8],
#        [ 7,  6,  5,  4],
#        [ 3,  2,  1,  0]])
```

## Indexing and slicing

`All` is numpy's bare `:` (Nim will not let `_` be an identifier). An integer
**drops** its axis; a span keeps it. Nim slices are inclusive at both ends,
and `^k` counts from the end.

```nim
let a = arange(12).reshape(3, 4)

echo a[1, 2]                               # an element
# 6

echo a[-1, -1]                             # negative indices count from the end
# 11

echo a[1, All]                             # a row: 1-d
# array([4, 5, 6, 7])

echo a[All, 1]                             # a column: 1-d
# array([1, 5, 9])

echo a[1..1, All]                          # the same row, still 2-d
# array([[4, 5, 6, 7]])

echo a[0..1, 1..2]
# array([[1, 2],
#        [5, 6]])

echo a[0, 1..^2]
# array([1, 2])

echo a[All, span(0, 3, 2)]                 # every other column
# array([[ 0,  2],
#        [ 4,  6],
#        [ 8, 10]])

echo a[0, span(3, 0, -1)]                  # reversed
# array([3, 2, 1, 0])

echo a.row(2)
# array([ 8,  9, 10, 11])

echo a.col(0)
# array([0, 4, 8])
```

Assignment takes the same selections. A scalar fills; an array broadcasts:

```nim
var a = zeros[int](3, 4)
a[0, 0] = 1
a[1, All] = 7
a[All, 3] = toNDArray(@[1, 2, 3])

echo a
# array([[1, 0, 0, 1],
#        [7, 7, 7, 2],
#        [0, 0, 0, 3]])

a[0..1, 0..1] = ones[int](2, 2)

echo a
# array([[1, 1, 0, 1],
#        [1, 1, 7, 2],
#        [0, 0, 0, 3]])
```

## Masks, `where` and `take`

Comparisons give an `NDArray[bool]`; indexing with one selects the elements
it marks, flattened.

```nim
let a = arange(12).reshape(3, 4)

echo a > 6
# array([[false, false, false, false],
#        [false, false, false,  true],
#        [ true,  true,  true,  true]])

echo a[a > 6]
# array([ 7,  8,  9, 10, 11])

echo a[(a mod 3).eq(0)]
# array([0, 3, 6, 9])

echo (a > 2) and (a < 8)
# array([[false, false, false,  true],
#        [ true,  true,  true,  true],
#        [false, false, false, false]])

echo where((a mod 2).eq(0), a, a * -1)
# array([[  0,  -1,   2,  -3],
#        [  4,  -5,   6,  -7],
#        [  8,  -9,  10, -11]])

echo sum(a > 6)                            # summing a mask counts
# 5

echo sum(a > 6, axis = 1)
# array([0, 1, 4])
```

Masked assignment leaves the rest alone, and `take` gathers along an axis in
any order, repeats allowed:

```nim
let a = arange(12).reshape(3, 4)
var b = arange(6)
b[b > 3] = 0

echo b
# array([0, 1, 2, 3, 0, 0])

echo (arange(5) > 2).nonZero
# @[3, 4]

echo a.take(@[2, 0], axis = 0)
# array([[ 8,  9, 10, 11],
#        [ 0,  1,  2,  3]])

echo a.take(@[3, 3], axis = 1)
# array([[ 3,  3],
#        [ 7,  7],
#        [11, 11]])
```

## Elementwise operations

```nim
let a = toNDArray(@[1.0, 2.0, 3.0])

echo a + a
# array([2.0, 4.0, 6.0])

echo a * 2.0
# array([2.0, 4.0, 6.0])

echo 10.0 - a
# array([9.0, 8.0, 7.0])

echo a / 2.0
# array([0.5, 1.0, 1.5])

echo -a
# array([-1.0, -2.0, -3.0])
```

`/` on integers gives floats, as numpy's does; `div` is the truncating one:

```nim
echo arange(5) / 2
# array([0.0, 0.5, 1.0, 1.5, 2.0])

echo arange(5) div 2
# array([0, 0, 1, 1, 2])

echo arange(5) mod 2
# array([0, 1, 0, 1, 0])
```

```nim
echo sqrt(toNDArray(@[1.0, 4.0, 9.0]))
# array([1.0, 2.0, 3.0])

echo exp(toNDArray(@[0.0, 1.0]))
# array([    1.0, 2.71828])

echo abs(toNDArray(@[-1, 2, -3]))
# array([1, 2, 3])

echo pow(toNDArray(@[2.0, 3.0]), 3.0)
# array([ 8.0, 27.0])

echo clip(arange(6), 1, 4)
# array([1, 1, 2, 3, 4, 4])

echo maximum(toNDArray(@[1, 5, 3]), toNDArray(@[4, 2, 3]))    # elementwise
# array([4, 5, 3])

echo minimum(arange(5), 2)
# array([0, 1, 2, 2, 2])

echo sign(toNDArray(@[-2.0, 0.0, 4.0]))
# array([-1.0,  0.0,  1.0])
```

Also here: `ln`, `log2`, `log10`, the trig and hyperbolic families,
`floor`/`ceil`/`round`/`trunc`, `square`, `reciprocal`, `degToRad`,
`isNaN`, `isFinite`, `allclose`.

## Broadcasting

The numpy rule: line the shapes up at their **trailing** axes, and each pair
must be equal or one of them 1.

```nim
let a = arange(6).reshape(2, 3)

echo a + arange(3)                         # (2,3) + (3,)
# array([[0, 2, 4],
#        [3, 5, 7]])

echo a + arange(2).reshape(2, 1)           # (2,3) + (2,1)
# array([[0, 1, 2],
#        [4, 5, 6]])

echo arange(3).reshape(3, 1) + arange(4).reshape(1, 4)
# array([[0, 1, 2, 3],
#        [1, 2, 3, 4],
#        [2, 3, 4, 5]])

echo arange(3).broadcastTo(@[2, 3]).describe
# NDArray[int] shape=(2, 3) strides=(0, 1) strided
```

A stretched axis gets a **stride of 0** — it repeats one element rather than
copying it, which is why broadcasting is free and why you should not write
through a broadcast view.

## Writing your own ufuncs

`mapIt` and `zipIt` are what every operator above is built from. The result's
element type is whatever the expression returns, so a predicate gives a mask
with no extra machinery:

```nim
let a = arange(5)

echo a.mapIt(it * it)
# array([ 0,  1,  4,  9, 16])

echo a.mapIt(it.float / 2.0)
# array([0.0, 0.5, 1.0, 1.5, 2.0])

echo a.mapIt(it mod 2 == 0)
# array([ true, false,  true, false,  true])

echo zipIt(arange(4), arange(4) * 2, x + y)   # zipIt sees x and y
# array([0, 3, 6, 9])

var b = arange(4)
b.applyIt(it * 100)                           # in place, through the view

echo b
# array([  0, 100, 200, 300])
```

## Reductions

Every fold works on the whole array, or along one axis. An axis reduction
removes that axis; `keepDims = true` leaves it at length 1, which is what
makes the result broadcast back over the original.

```nim
let a = arange(12).reshape(3, 4)

echo sum(a)
# 66

echo mean(a)
# 5.5

echo min(a), " ", max(a), " ", ptp(a)
# 0 11 11

echo argmax(a)                             # a flat index
# 11
```

```nim
let a = arange(12).reshape(3, 4)

echo sum(a, axis = 0)
# array([12, 15, 18, 21])

echo sum(a, axis = 1)
# array([ 6, 22, 38])

echo mean(a, axis = 0)
# array([4.0, 5.0, 6.0, 7.0])

echo max(a, axis = 1)
# array([ 3,  7, 11])

echo argmax(a, axis = 1)                   # positions along the axis
# array([3, 3, 3])

echo mean(a, axis = 1, keepDims = true)
# array([[1.5],
#        [5.5],
#        [9.5]])

echo a.astype(float) - mean(a, axis = 1, keepDims = true)
# array([[-1.5, -0.5,  0.5,  1.5],
#        [-1.5, -0.5,  0.5,  1.5],
#        [-1.5, -0.5,  0.5,  1.5]])
```

`std` and `variance` have both a whole-array and an axis form, so their
second argument must be **named**:

```nim
let a = arange(12).reshape(3, 4)

echo std(a, ddof = 1)
# 3.605551275463989

echo std(a, axis = 0)
# array([3.26599, 3.26599, 3.26599, 3.26599])

echo variance(toNDArray(@[1.0, 2.0, 3.0]), ddof = 1)
# 1.0
```

Booleans and scans:

```nim
let a = arange(12).reshape(3, 4)

echo all(a >= 0)
# true

echo any(a > 10)
# true

echo all(a > 3, axis = 1)
# array([false,  true,  true])

echo cumsum(arange(6))
# array([ 0,  1,  3,  6, 10, 15])

echo cumprod(arange(1, 6))
# array([  1,   2,   6,  24, 120])

echo cumsum(arange(6).reshape(2, 3), axis = 1)
# array([[ 0,  1,  3],
#        [ 3,  7, 12]])

echo diff(toNDArray(@[1, 4, 9, 16]))
# array([3, 5, 7])
```

### Data with holes in it

An `NDArray` has no notion of a missing value — that is racoon's job, and the
bridge turns an NA into a NaN on the way over. A NaN then poisons an ordinary
fold, which is usually what you want to be told:

```nim
let h = toNDArray(@[@[1.0, NaN, 3.0], @[NaN, NaN, 6.0]])

echo mean(h)
# nan
```

The `nan`-prefixed folds skip them instead, and `nanCount` says how much of
the answer is real:

```nim
echo nanMean(h)
# 3.3333333333333335

echo nanCount(h, axis = 0)
# array([1, 0, 2])

echo nanMean(h, axis = 1)
# array([2.0, 6.0])
```

A slice with nothing left in it has no mean, and the two forms disagree about
that on purpose. `nanMean(allNaN)` raises, as `mean` of an empty array does.
Along an axis it gives NaN for that slice only, since raising would throw away
the answers for every other row:

```nim
echo nanMean(h, axis = 0)
# array([1.0, nan, 4.5])
```

Nim reads `nansum` and `nanSum` as the same identifier, so numpy's spelling
works unchanged. `nanVariance` is spelled out, following `variance`, and it
takes the same care: write `nanStd(a, axis = 0)` or `nanStd(a, ddof = 1)`,
never a bare second argument.

## Linear algebra

`*` is elementwise, as in numpy, so the matrix product is `matmul`.

```nim
let a = arr(@[@[1.0, 2.0], @[3.0, 4.0]])
let b = arr(@[@[5.0, 6.0], @[7.0, 8.0]])

echo matmul(a, b)
# array([[19.0, 22.0],
#        [43.0, 50.0]])

echo matmul(a, toNDArray(@[1.0, 1.0]))     # a 1-d operand is a column
# array([3.0, 7.0])

echo dot(toNDArray(@[1.0, 2.0, 3.0]), toNDArray(@[4.0, 5.0, 6.0]))
# 32.0

echo outer(toNDArray(@[1.0, 2.0]), toNDArray(@[3.0, 4.0]))
# array([[3.0, 4.0],
#        [6.0, 8.0]])

echo trace(a)
# 5.0

echo diag(a)                               # matrix -> its diagonal
# array([1.0, 4.0])

echo diag(toNDArray(@[1.0, 2.0]))          # vector -> a matrix
# array([[1.0, 0.0],
#        [0.0, 2.0]])

echo norm(toNDArray(@[3.0, 4.0]))
# 5.0
```

`solve`, `inv` and `det` are three readings of one LU factorisation with
partial pivoting:

```nim
let a = arr(@[@[1.0, 2.0], @[3.0, 4.0]])

echo det(a)
# -2.0

echo inv(a)
# array([[-2.0,  1.0],
#        [ 1.5, -0.5]])

echo solve(a, toNDArray(@[5.0, 11.0]))
# array([1.0, 2.0])
```

`lstsq` fits an overdetermined system — here a straight line, by putting a
column of ones next to x:

```nim
let x = toNDArray(@[0.0, 1.0, 2.0, 3.0, 4.0])
let y = x * 2.0 + 1.0
let design = hstack(@[x.reshape(5, 1), ones[float](5, 1)])

echo design
# array([[0.0, 1.0],
#        [1.0, 1.0],
#        [2.0, 1.0],
#        [3.0, 1.0],
#        [4.0, 1.0]])

echo lstsq(design, y)                      # slope, intercept
# array([2.0, 1.0])
```

## Statistics

```nim
let a = toNDArray(@[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

echo sortedArray(a)
# array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])

echo argsort(a)                            # the permutation, not the values
# array([1, 3, 6, 0, 2, 4, 7, 5])

echo median(a)
# 3.5

echo quantile(a, 0.25)
# 1.75

echo percentile(a, 90.0)
# 6.8999999999999995

echo iqr(a)
# 3.5

echo unique(toNDArray(@[3, 1, 3, 1, 2]))
# array([1, 2, 3])

echo bincount(toNDArray(@[0, 1, 1, 3, 3, 3]))
# array([1, 2, 0, 3])
```

### Holes in the data

NaN sorts after every number, as it does in numpy, and the order statistics
propagate it — where a hole would sit in the ordering is exactly what decides
the middle, so `median` cannot ignore one and still be honest. `nanMedian`
and `nanQuantile` are the forms that drop holes and answer over the rest,
the same split as `mean` and `nanMean`:

```nim
let a = toNDArray(@[3.0, 1.0, NaN, 5.0, 2.0])

echo sortedArray(a)
# array([1.0, 2.0, 3.0, 5.0, nan])

echo argsort(a)                            # the hole's index goes last
# array([1, 4, 0, 3, 2])

echo median(a)
# nan

echo nanMedian(a)
# 2.5

echo nanQuantile(a, 0.25)
# 1.75

echo unique(toNDArray(@[3.0, NaN, 1.0, NaN, 3.0]))   # one NaN, at the end
# array([1.0, 3.0, nan])
```

`histogram` returns the counts and the `bins + 1` edges that produced them:

```nim
let a = toNDArray(@[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
let (counts, edges) = histogram(a, bins = 4)

echo counts
# array([3, 2, 2, 1])

echo edges
# array([1.0, 3.0, 5.0, 7.0, 9.0])

echo normalize(a)                          # rescaled to 0..1
# array([ 0.25,   0.0, 0.375,   0.0,   0.5,   1.0, 0.125, 0.625])

echo zscore(a)
# array([-0.340352,   -1.1183, 0.0486217,   -1.1183,  0.437595,   1.99349, -0.729325,  0.826568])

echo corr(a, a * 2.0)
# 1.0
```

The range is the data's own span unless you ask for one, and you ask with an
`Option` — a tuple has no spare value to mean "not given", so the interval
from 0 to 0 would otherwise be unaskable. A hole is not a value and lands in
no bin, so `sum(counts)` is `nanCount(a)` when the data has holes in it:

```nim
import std/options

let a = toNDArray(@[0.0, 5.0, 10.0, NaN])

echo histogram(a, bins = 2, range = some((0.0, 6.0)))[0]   # 10.0 is outside
# array([1, 1])

echo histogram(a, bins = 2)[0]             # NaN reaches neither bin nor edge
# array([1, 2])

echo histogram(a, bins = 2)[1]
# array([ 0.0,  5.0, 10.0])
```

## Random

The generator is `std/random`'s global one, so these are `proc`, not `func`.
Its default state is fixed — `seed` makes a run reproducible, `randomSeed()`
asks for a different one each time.

```nim
seed(42)

echo randArray(2, 3)                       # uniform on [0, 1)
# array([[0.711028, 0.437956, 0.124115],
#        [0.379889, 0.750551, 0.170067]])

echo randint(0, 10, 8)
# array([7, 7, 6, 8, 6, 0, 2, 0])

echo uniform(-1.0, 1.0, 4)
# array([0.525056, 0.212118, -0.30394, 0.585824])

echo permutation(8)
# array([2, 6, 5, 1, 4, 0, 7, 3])

echo shuffled(arange(6))
# array([4, 3, 2, 0, 1, 5])

echo choice(arange(10), 4, replace = false)
# array([0, 6, 9, 3])
```

```nim
seed(42)
let n = randn(10000)                       # standard normal

echo mean(n)
# 0.00918817015996703

echo std(n)
# 0.9880367604988528
```

## With racoon

A racoon column comes across as a 1-d array, and a frame as a design matrix.

```nim
import num
import num/racoon
import racoon

let df = """plate,well,signal,control
p1,A01,1.2,true
p1,A02,3.4,false
p1,A03,,false
p2,A01,5.6,true
p2,A02,7.8,false""".toDataFrame()

echo df
# +-------+-------+---------+---------+
# | plate |  well |  signal | control |
# | <str> | <str> | <float> |  <bool> |
# +-------+-------+---------+---------+
# |    p1 |   A01 |     1.2 |    true |
# |    p1 |   A02 |     3.4 |   false |
# |    p1 |   A03 |      NA |   false |
# |    p2 |   A01 |     5.6 |    true |
# |    p2 |   A02 |     7.8 |   false |
# +-------+-------+---------+---------+
# shape = [5, 4]

echo df["signal"].toArray()
# array([1.2, 3.4, nan, 5.6, 7.8])

echo df.toMatrix(@["signal"]).describe
# NDArray[float] shape=(5, 1) strides=(1, 1) contiguous
```

Racoon's NA has no counterpart in an array, so it arrives as **NaN** — and
NaN propagates, which is the honest answer until you say what to do about it:

```nim
import num
import num/racoon
import racoon

let df = """well,signal
A01,1.2
A02,3.4
A03,
A04,5.6""".toDataFrame()
let s = df["signal"].toArray()

echo s.mean()
# nan

echo s.isNaN
# array([false, false,  true, false])

echo s[s.isNaN.not]
# array([1.2, 3.4, 5.6])

echo mean(s[s.isNaN.not])
# 3.4
```

Going the other way, a NaN stays an ordinary value unless you say it came
from an NA — racoon treats NaN as a value, and this bridge does not
contradict it:

```nim
import num
import num/racoon
import racoon

let df = """well,signal
A01,1.2
A02,3.4
A03,
A04,5.6""".toDataFrame()
let s = df["signal"].toArray()

echo df.addColumn(toColumn("scaled", normalize(s), nanAsNA = true))
# +-------+---------+--------------------+
# |  well |  signal |             scaled |
# | <str> | <float> |            <float> |
# +-------+---------+--------------------+
# |   A01 |     1.2 |                0.0 |
# |   A02 |     3.4 | 0.5000000000000001 |
# |   A03 |      NA |                 NA |
# |   A04 |     5.6 |                1.0 |
# +-------+---------+--------------------+
# shape = [4, 3]

echo toDataFrame(arange(6).reshape(3, 2).astype(float), @["x", "y"])
# +---------+---------+
# |       x |       y |
# | <float> | <float> |
# +---------+---------+
# |     0.0 |     1.0 |
# |     2.0 |     3.0 |
# |     4.0 |     5.0 |
# +---------+---------+
# shape = [3, 2]
```

For int and bool columns there is no NaN to fall back on, so a missing cell
raises unless you say what to put there: `col.toIntArray(fill = some(0))`.

## With plotter

The bridge adds plotter's own convenience constructors over arrays, so a
chart is one line from the data:

```nim
import num
import num/plotting
import plotter
import std/json

let x = linspace(0.0, 1.0, 5)
let c = scatter(x, x * x, "squares")

echo c.toSpec()["mark"]
# "point"

echo c.toSpec()["data"]["values"]
# [{"x":0.0,"y":0.0},{"x":0.25,"y":0.0625},{"x":0.5,"y":0.25},{"x":0.75,"y":0.5625},{"x":1.0,"y":1.0}]

echo c.toSpec()["encoding"]["y"]
# {"field":"y","type":"quantitative"}
```

From there it is plotter's own API — `save`, `show`, `properties`, and the
rest:

```nim
import num
import num/plotting
import plotter

let x = linspace(0.0, 6.28, 200)
line(x, sin(x), "sine").save("sine.png")

heatmap(randArray(8, 8)).show()

# several runs on one chart, the grammar way: one long frame, colour by group
line(@[toSeries("a", x, sin(x)), toSeries("b", x, cos(x))], "two runs").show()
```

`hist` leaves the binning to Vega-Lite; `histChart` bins here with
`num.histogram` first, for when the bars have to match what the rest of an
analysis computed. Its bars are the bin centres:

```nim
import num
import num/plotting
import plotter
import std/json

seed(1)
let x = randn(500)
let (counts, edges) = histogram(x, bins = 5)

echo counts
# array([ 17, 106, 220, 142,  15])

echo edges
# array([-3.30292, -1.99765, -0.69238,  0.61289,  1.91816,  3.22343])

echo histChart(x, bins = 5).toSpec()["data"]["values"][0]   # a bar: centre, count
# {"x":-2.650286322022022,"y":17.0}
```

The module is `num/plotting`, not `num/plotter` — a Nim module cannot import
another of its own name.

## Two rules worth knowing

**An all-integer index must name every axis.** `a[1, 2]` is an element;
`a[1]` on a 2-d array is an *error*, not its first row. `int` converts to a
selection, so element access and slicing cannot be overloads of each other,
and whether an index names every axis is a runtime fact that cannot pick a
return type. Write `a[1, All]` or `a.row(1)`.

**`==` is whole-array equality**, returning a `bool`, so elementwise equality
is `eq`/`neq` — the same choice racoon makes for its columns. Nim cannot
overload on the return type, and `if a == b` is the more common thing to
want. Both read best in method-call form, since as infix words they bind more
loosely than arithmetic: `(a mod 3).eq(0)`.

```nim
let a = arange(4)

echo a == arange(4)                        # one bool: the whole array
# true

echo a.eq(2)                               # a mask
# array([false, false,  true, false])

echo a.neq(2)
# array([ true,  true, false,  true])
```

## What's here

| | |
|---|---|
| **create** | `toNDArray`/`arr`, `zeros`, `ones`, `full`, `empty`, `arange`, `linspace`, `logspace`, `eye`, `identity`, `meshgrid`, `zerosLike`, `onesLike`, `fullLike` |
| **shape** | `reshape`, `ravel`, `flatten`, `transpose`/`t`, `swapAxes`, `moveAxis`, `expandDims`, `squeeze`, `flip`, `broadcastTo`, `broadcastShapes`, `concat`, `stack`, `vstack`, `hstack`, `astype` |
| **index** | `a[i, j]`, `a[1, All]`, `a[0..2, 1..^1]`, `span(0, 8, 2)`, `a[mask]`, `take`, `nonZero`, `row`, `col`, and the assigning form of each |
| **ops** | `+ - * /`, `div`, `mod`, `+=`/`-=`/`*=`/`/=`, comparisons → `NDArray[bool]`, `eq`/`neq`, `and`/`or`/`xor`/`not`, `sqrt`/`exp`/`ln`/trig/`floor`…, `abs`, `sign`, `pow`, `clip`, `maximum`/`minimum`, `where`, `allclose`, `mapIt`/`zipIt`/`applyIt` |
| **reduce** | `sum`, `prod`, `min`, `max`, `ptp`, `argmin`, `argmax`, `mean`, `variance`, `std`, `all`, `any`, `countNonZero`, `cumsum`, `cumprod`, `diff` — each whole-array or along an axis; `nanSum`, `nanMean`, `nanMin`, `nanMax`, `nanVariance`, `nanStd`, `nanCount` for data with holes in it |
| **linalg** | `dot`, `matmul`, `outer`, `trace`, `diag`, `norm`, `solve`, `inv`, `det`, `lstsq` |
| **stats** | `sortedArray`, `argsort`, `median`, `quantile`, `quantiles`, `percentile`, `iqr`, `unique`, `nanMedian`, `nanQuantile`, `bincount`, `histogram`, `cov`, `corr`, `normalize`, `zscore` |
| **random** | `seed`, `randomSeed`, `randArray`, `uniform`, `randn`, `normal`, `randint`, `choice`, `shuffled`, `permutation` |
| **inspect** | `shape`, `strides`, `ndim`, `size`, `len`, `isContiguous`, `dtype`, `describe`, `$`, `toSeq`, `item` |

Floats print to six significant digits, as numpy's `repr` does, and an array
of more than 1000 elements prints only its corners; `toSeq` and `$` on a
single element give you the full value either way.

## Tests

```sh
nimble test                  # every tests/t*.nim
nimble racoontest            # the racoon bridge, for real
nimble plottertest           # the plotter bridge, for real

RACOON_PATH=/path/to/Racoon/src nimble racoontest
PLOTTER_PATH=/path/to/nim-plotter/src nimble plottertest
```

The two bridge suites are behind `when defined(racoon)` / `when
defined(plotter)`, so `nimble test` still compiles them with neither package
installed.

## Licence

MIT.
