## Folds: over the whole array, or along one axis.
##
## Every axis reduction is written with the `axisFold` template, so the
## traversal, the output shape and `keepDims` are each stated once and a new
## reduction is a line. The whole-array forms are separate rather than
## `axisFold` over a flattened copy — they need no allocation at all.
##
## An axis reduction **removes** that axis (`keepDims = true` leaves it at
## length 1), which is what makes `a - a.mean(axis = 1, keepDims = true)`
## broadcast back over the rows.

import std/[math, algorithm]
import ./core
import ./shape
import ./creation

template axisFold*(a: untyped, axisArg: int, keepDims: bool,
                   initVal, accExpr: untyped): untyped =
  ## Fold along one axis. `accExpr` sees the running accumulator as `acc` and
  ## the element as `it`; the result's element type is `initVal`'s.
  block:
    let src = a
    let ax = normAxis(axisArg, src.ndim)
    var redShape = src.shape
    redShape.delete(ax)
    let rstr = contiguousStrides(redShape)
    var res = full(redShape, initVal)
    for idx, elem in src.pairs:
      var o = 0
      for k in 0 ..< src.ndim:
        if k < ax: o += idx[k] * rstr[k]
        elif k > ax: o += idx[k] * rstr[k - 1]
      let acc {.inject.} = res.buf[o]
      let it {.inject.} = elem
      res.buf[o] = accExpr
    if keepDims:
      var keepShape = src.shape
      keepShape[ax] = 1
      res = res.reshape(keepShape)
    res

# ----------------------------------------------------------------- sums ----

proc sum*[T](a: NDArray[T]): T =
  result = T(0)
  for x in a: result = result + x

proc sum*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  axisFold(a, axis, keepDims, T(0), acc + it)

proc prod*[T](a: NDArray[T]): T =
  result = T(1)
  for x in a: result = result * x

proc prod*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  axisFold(a, axis, keepDims, T(1), acc * it)

# ------------------------------------------------------------ extremes ----

func checkNonEmpty[T](a: NDArray[T], op: string) =
  ## An empty fold has no answer for min/max — there is no identity to
  ## return — so it raises rather than inventing one.
  if a.size == 0:
    raise newException(ValueError, op & " of an empty array")

proc min*[T](a: NDArray[T]): T =
  checkNonEmpty(a, "min")
  var first = true
  for x in a:
    if first or x < result:
      result = x
      first = false

proc max*[T](a: NDArray[T]): T =
  checkNonEmpty(a, "max")
  var first = true
  for x in a:
    if first or x > result:
      result = x
      first = false

proc min*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  checkNonEmpty(a, "min")
  axisFold(a, axis, keepDims, high(T), system.min(acc, it))

proc max*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  checkNonEmpty(a, "max")
  axisFold(a, axis, keepDims, low(T), system.max(acc, it))

proc ptp*[T](a: NDArray[T]): T =
  ## Peak to peak: `max - min`, numpy's name for the range.
  max(a) - min(a)

proc argmin*[T](a: NDArray[T]): int =
  ## Flat index of the smallest element, ties going to the first.
  checkNonEmpty(a, "argmin")
  var best: T
  var i = 0
  for x in a:
    if i == 0 or x < best:
      best = x
      result = i
    i.inc

proc argmax*[T](a: NDArray[T]): int =
  checkNonEmpty(a, "argmax")
  var best: T
  var i = 0
  for x in a:
    if i == 0 or x > best:
      best = x
      result = i
    i.inc

proc argExtreme[T](a: NDArray[T], axis: int, keepDims: bool,
                   wantMax: bool): NDArray[int] =
  ## `argmin`/`argmax` along an axis. Not an `axisFold`: the accumulator is a
  ## pair (best value, its position) and only the position is returned.
  checkNonEmpty(a, "arg reduction")
  let ax = normAxis(axis, a.ndim)
  var redShape = a.shape
  redShape.delete(ax)
  let rstr = contiguousStrides(redShape)
  var best = newSeq[T](prod(redShape))
  var seen = newSeq[bool](prod(redShape))
  result = newNDArray[int](redShape)
  for idx, elem in a.pairs:
    var o = 0
    for k in 0 ..< a.ndim:
      if k < ax: o += idx[k] * rstr[k]
      elif k > ax: o += idx[k] * rstr[k - 1]
    if not seen[o] or (if wantMax: elem > best[o] else: elem < best[o]):
      seen[o] = true
      best[o] = elem
      result.buf[o] = idx[ax]
  if keepDims:
    var keepShape = a.shape
    keepShape[ax] = 1
    result = result.reshape(keepShape)

proc argmin*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[int] =
  argExtreme(a, axis, keepDims, false)
proc argmax*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[int] =
  argExtreme(a, axis, keepDims, true)

# ------------------------------------------------------------- moments ----

proc mean*[T](a: NDArray[T]): float =
  ## Always a float, whatever the element type — the same rule racoon's
  ## `mean` follows, and for the same reason: it is a fold of `/`.
  checkNonEmpty(a, "mean")
  var s = 0.0
  for x in a: s += float(x)
  s / float(a.size)

proc mean*[T](a: NDArray[T], axis: int, keepDims = false): NDArray[float] =
  checkNonEmpty(a, "mean")
  let n = float(a.shape[normAxis(axis, a.ndim)])
  let totals = axisFold(a, axis, keepDims, 0.0, acc + float(it))
  for i in 0 ..< totals.buf[].len: totals.buf[i] = totals.buf[i] / n
  totals

proc variance*[T](a: NDArray[T], ddof = 0): float =
  ## Population variance by default; `ddof = 1` for the sample one. Two
  ## passes rather than the sum-of-squares shortcut, which loses most of its
  ## precision when the mean is large relative to the spread.
  if a.size <= ddof:
    raise newException(ValueError, "variance needs more than " & $ddof & " elements")
  let m = mean(a)
  var s = 0.0
  for x in a:
    let d = float(x) - m
    s += d * d
  s / float(a.size - ddof)

proc std*[T](a: NDArray[T], ddof = 0): float = sqrt(variance(a, ddof = ddof))

proc variance*[T](a: NDArray[T], axis: int, ddof = 0, keepDims = false): NDArray[float] =
  let n = a.shape[normAxis(axis, a.ndim)]
  if n <= ddof:
    raise newException(ValueError, "variance needs more than " & $ddof & " elements")
  let m = mean(a, axis, keepDims = true)
  let mb = m.broadcastTo(a.shape)
  var sq = newNDArray[float](a.shape)
  var i = 0
  for oa, ob in offsets2(a, mb):
    let d = float(a.buf[oa]) - mb.buf[ob]
    sq.buf[i] = d * d
    i.inc
  result = sum(sq, axis, keepDims)
  for j in 0 ..< result.buf[].len:
    result.buf[j] = result.buf[j] / float(n - ddof)

proc std*[T](a: NDArray[T], axis: int, ddof = 0, keepDims = false): NDArray[float] =
  result = variance(a, axis, ddof, keepDims)
  for i in 0 ..< result.buf[].len: result.buf[i] = sqrt(result.buf[i])

# --------------------------------------------------------- NaN-aware ----

## An `NDArray` has no validity mask — holes belong to racoon's frame, and
## the bridge maps NA to NaN on the way over. These are what makes that
## survivable: the same folds with NaN skipped rather than propagated.
##
## Where there is nothing left to fold, the two forms differ deliberately.
## The whole-array form raises, like `mean` of an empty array does — there
## is no answer and a caller asking for one has a bug. The axis form returns
## NaN for that slice, because raising would throw away the answers for
## every other slice on account of one, and NaN is a float the result can
## hold. numpy does the same.
##
## Nim reads `nansum` and `nanSum` as one identifier, so numpy's spelling
## works unchanged. `nanVariance` follows `variance` in being spelled out.

proc nanCount*[T: SomeFloat](a: NDArray[T]): int =
  ## How many elements are not NaN — how much of a result is real.
  for x in a:
    if not isNaN(x): result.inc

proc nanCount*[T: SomeFloat](a: NDArray[T], axis: int, keepDims = false): NDArray[int] =
  axisFold(a, axis, keepDims, 0, (if isNaN(it): acc else: acc + 1))

proc nanSum*[T: SomeFloat](a: NDArray[T]): T =
  ## Sum of the non-NaN elements. All-NaN gives 0, the identity — as with
  ## `sum` of an empty array, and as numpy gives.
  result = T(0)
  for x in a:
    if not isNaN(x): result = result + x

proc nanSum*[T: SomeFloat](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  axisFold(a, axis, keepDims, T(0), (if isNaN(it): acc else: acc + it))

func checkAnyValid(n: int, op: string) =
  if n == 0:
    raise newException(ValueError, op & " of an array with no non-NaN values")

proc nanMin*[T: SomeFloat](a: NDArray[T]): T =
  checkAnyValid(a.nanCount, "nanMin")
  var first = true
  for x in a:
    if not isNaN(x) and (first or x < result):
      result = x
      first = false

proc nanMax*[T: SomeFloat](a: NDArray[T]): T =
  checkAnyValid(a.nanCount, "nanMax")
  var first = true
  for x in a:
    if not isNaN(x) and (first or x > result):
      result = x
      first = false

proc nanExtreme[T: SomeFloat](a: NDArray[T], axis: int, keepDims: bool,
                              wantMax: bool): NDArray[T] =
  ## The counts decide which slices are empty rather than a sentinel in the
  ## accumulator: an array whose values really are all `Inf` would be
  ## indistinguishable from an all-NaN slice otherwise.
  result =
    if wantMax: axisFold(a, axis, keepDims, T(NegInf),
                         (if isNaN(it): acc else: system.max(acc, it)))
    else: axisFold(a, axis, keepDims, T(Inf),
                   (if isNaN(it): acc else: system.min(acc, it)))
  let counts = nanCount(a, axis, keepDims)
  for i in 0 ..< result.buf[].len:
    if counts.buf[i] == 0: result.buf[i] = T(NaN)

proc nanMin*[T: SomeFloat](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  nanExtreme(a, axis, keepDims, false)

proc nanMax*[T: SomeFloat](a: NDArray[T], axis: int, keepDims = false): NDArray[T] =
  nanExtreme(a, axis, keepDims, true)

proc nanMean*[T: SomeFloat](a: NDArray[T]): float =
  let n = a.nanCount
  checkAnyValid(n, "nanMean")
  var s = 0.0
  for x in a:
    if not isNaN(x): s += float(x)
  s / float(n)

proc nanMean*[T: SomeFloat](a: NDArray[T], axis: int, keepDims = false): NDArray[float] =
  let totals = axisFold(a, axis, keepDims, 0.0,
                        (if isNaN(it): acc else: acc + float(it)))
  let counts = nanCount(a, axis, keepDims)
  result = totals
  for i in 0 ..< result.buf[].len:
    result.buf[i] =
      if counts.buf[i] == 0: NaN else: result.buf[i] / float(counts.buf[i])

proc nanVariance*[T: SomeFloat](a: NDArray[T], ddof = 0): float =
  ## Two passes over the non-NaN elements, as `variance` is over all of them.
  let n = a.nanCount
  if n <= ddof:
    raise newException(ValueError,
      "nanVariance needs more than " & $ddof & " non-NaN elements")
  let m = nanMean(a)
  var s = 0.0
  for x in a:
    if not isNaN(x):
      let d = float(x) - m
      s += d * d
  s / float(n - ddof)

proc nanStd*[T: SomeFloat](a: NDArray[T], ddof = 0): float =
  sqrt(nanVariance(a, ddof = ddof))

proc nanVariance*[T: SomeFloat](a: NDArray[T], axis: int, ddof = 0,
                                keepDims = false): NDArray[float] =
  ## A slice with no more than `ddof` real values gives NaN rather than
  ## raising, for the reason at the top of this section.
  let m = nanMean(a, axis, keepDims = true)
  let mb = m.broadcastTo(a.shape)
  var sq = newNDArray[float](a.shape)
  var i = 0
  for oa, ob in offsets2(a, mb):
    let x = a.buf[oa]
    sq.buf[i] = if isNaN(x): NaN else: (float(x) - mb.buf[ob]) * (float(x) - mb.buf[ob])
    i.inc
  result = nanSum(sq, axis, keepDims)
  let counts = nanCount(a, axis, keepDims)
  for j in 0 ..< result.buf[].len:
    let n = counts.buf[j]
    result.buf[j] = if n <= ddof: NaN else: result.buf[j] / float(n - ddof)

proc nanStd*[T: SomeFloat](a: NDArray[T], axis: int, ddof = 0,
                           keepDims = false): NDArray[float] =
  ## Like `std`, the whole-array and axis forms both take an `int` second
  ## argument, so name it: `nanStd(a, axis = 0)` or `nanStd(a, ddof = 1)`.
  result = nanVariance(a, axis, ddof, keepDims)
  for i in 0 ..< result.buf[].len: result.buf[i] = sqrt(result.buf[i])

# ------------------------------------------------------------ booleans ----

proc all*(a: NDArray[bool]): bool =
  for x in a:
    if not x: return false
  true

proc any*(a: NDArray[bool]): bool =
  for x in a:
    if x: return true
  false

proc all*(a: NDArray[bool], axis: int, keepDims = false): NDArray[bool] =
  axisFold(a, axis, keepDims, true, acc and it)

proc any*(a: NDArray[bool], axis: int, keepDims = false): NDArray[bool] =
  axisFold(a, axis, keepDims, false, acc or it)

proc countNonZero*(a: NDArray[bool]): int =
  for x in a:
    if x: result.inc

proc countNonZero*[T](a: NDArray[T]): int =
  for x in a:
    if x != T(0): result.inc

proc sum*(a: NDArray[bool]): int =
  ## Summing a mask counts its `true`s — `bool` has no `+`, and counting is
  ## the only thing the fold could mean. Same answer as `countNonZero`,
  ## under the name a numpy reader reaches for first.
  for x in a:
    if x: result.inc

proc sum*(a: NDArray[bool], axis: int, keepDims = false): NDArray[int] =
  ## Counting `true`s, which is what summing a mask means.
  axisFold(a, axis, keepDims, 0, acc + (if it: 1 else: 0))

# ----------------------------------------------------------- scans --------

proc cumsum*[T](a: NDArray[T]): NDArray[T] =
  ## Running total over the flattened array, as numpy's axis-less form does.
  result = newNDArray[T](a.size)
  var acc = T(0)
  var i = 0
  for x in a:
    acc = acc + x
    result.buf[i] = acc
    i.inc

proc cumprod*[T](a: NDArray[T]): NDArray[T] =
  result = newNDArray[T](a.size)
  var acc = T(1)
  var i = 0
  for x in a:
    acc = acc * x
    result.buf[i] = acc
    i.inc

proc cumsum*[T](a: NDArray[T], axis: int): NDArray[T] =
  ## Running total along one axis; the shape is unchanged.
  let ax = normAxis(axis, a.ndim)
  result = a.copy()
  var idx = newSeq[int](a.ndim)
  for off in result.offsets:
    if idx[ax] > 0:
      result.buf[off] = result.buf[off - result.strides[ax]] + result.buf[off]
    for k in countdown(a.ndim - 1, 0):
      idx[k] += 1
      if idx[k] < a.shape[k]: break
      idx[k] = 0

proc diff*[T](a: NDArray[T]): NDArray[T] =
  ## First differences of a 1-d array; one element shorter, like numpy.
  if a.ndim != 1:
    raise newException(ValueError, "diff takes a 1-d array")
  let n = max(0, a.size - 1)
  result = newNDArray[T](n)
  let xs = a.toSeq()
  for i in 0 ..< n: result.buf[i] = xs[i + 1] - xs[i]
