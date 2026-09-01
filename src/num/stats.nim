## Order statistics, distributions of values, and the two-variable summaries.
##
## The sorting-based ones flatten first: an axis-wise `median` would need a
## per-lane gather, and the 1-d case is what the plotting and summary code
## actually asks for. `sortedArray`/`argsort` are the exception — they are
## genuinely 1-d operations and say so.
##
## **NaN sorts last and propagates.** Everything here that orders values uses
## `nanLast` rather than `cmp`, and everything that reads an order statistic
## out of the result gives NaN if the input held one — see `nanLast` for why
## the first is not optional and `median` for why the second is the only
## honest answer. `nanMedian`/`nanQuantile` are the forms that skip holes,
## as `nanMean` is for `mean`.

import std/[algorithm, math, options]
import ./core
import ./shape
import ./creation
import ./index
import ./ops
import ./reductions

func nanLast*[T](x, y: T): int =
  ## The comparator every sort here uses: ordinary order, with NaN after
  ## every number. Nim's `cmp` is **not** a valid comparator for floats —
  ## NaN compares false to everything, so `cmp` calls it "greater" in both
  ## directions, and a sort given an inconsistent comparator returns an
  ## order that depends on where the NaN happened to sit. `median` of the
  ## same multiset then answered differently depending on the input order.
  ## Non-float types never reach the NaN branch: `when` compiles it away.
  when T is SomeFloat:
    let xn = isHole(x)
    let yn = isHole(y)
    if xn or yn:
      if xn and yn: 0 elif xn: 1 else: -1
    else:
      cmp(x, y)
  else:
    cmp(x, y)

func hasNaN*[T](a: NDArray[T]): bool =
  ## Whether any element is NaN. Always false for a non-float array.
  when T is SomeFloat:
    for x in a:
      if isHole(x): return true
  false

func withoutNaN[T: SomeFloat](a: NDArray[T]): seq[T] =
  ## The real values, in input order — what the `nan`-prefixed order
  ## statistics work on. A copy, like everything else that sorts.
  for x in a:
    if not isHole(x): result.add(x)

proc sortedArray*[T](a: NDArray[T], ascending = true): NDArray[T] =
  ## The elements in order, as a new 1-d array. Named `sortedArray` rather
  ## than `sort` because nothing here mutates — the same reason racoon calls
  ## its frame version `sorted`.
  ##
  ## NaN sorts after every number, as numpy's `sort` puts it. `ascending =
  ## false` reverses the whole ordering, NaN included, so the holes come
  ## first — that is what numpy's `sort(a)[::-1]` gives too.
  var xs = a.toSeq()
  xs.sort(nanLast[T])
  if not ascending: xs.reverse()
  initNDArray[T](@[xs.len], xs)

proc argsort*[T](a: NDArray[T], ascending = true): NDArray[int] =
  ## The permutation that would sort `a` — the indices, not the values, so
  ## it can be fed to `take` to reorder something else alongside. Stable, so
  ## ties keep their input order in both directions.
  let xs = a.toSeq()
  var idx = newSeq[int](xs.len)
  for i in 0 ..< xs.len: idx[i] = i
  idx.sort(proc (i, j: int): int =
    let c = nanLast(xs[i], xs[j])
    if c != 0: (if ascending: c else: -c) else: cmp(i, j))
  initNDArray[int](@[idx.len], idx)

proc median*[T](a: NDArray[T]): float =
  ## The middle value, averaging the two middles for an even count.
  ##
  ## NaN propagates: a hole anywhere makes the answer NaN, as it does for
  ## `mean` and `sum`. It cannot be ignored quietly — where the hole sits in
  ## the sorted order is exactly what decides the middle. `nanMedian` is the
  ## form that drops holes and answers over what is left.
  if a.size == 0: raise newException(ValueError, "median of an empty array")
  if hasNaN(a): return NaN
  let xs = sortedArray(a).toSeq()
  let n = xs.len
  if n mod 2 == 1: float(xs[n div 2])
  else: (float(xs[n div 2 - 1]) + float(xs[n div 2])) / 2.0

proc quantile*[T](a: NDArray[T], q: float): float =
  ## Linear interpolation between the two neighbouring order statistics —
  ## numpy's default `method="linear"`, so `quantile(x, 0.5) == median(x)`.
  if a.size == 0: raise newException(ValueError, "quantile of an empty array")
  if q < 0.0 or q > 1.0:
    raise newException(ValueError, "quantile: q must be in 0..1, got " & $q)
  if hasNaN(a): return NaN          # as `median`, and for the same reason
  let xs = sortedArray(a).toSeq()
  let pos = q * float(xs.len - 1)
  let lo = int(floor(pos))
  let hi = int(ceil(pos))
  if lo == hi: float(xs[lo])
  else: float(xs[lo]) + (pos - float(lo)) * (float(xs[hi]) - float(xs[lo]))

proc percentile*[T](a: NDArray[T], p: float): float = quantile(a, p / 100.0)

proc quantiles*[T](a: NDArray[T], qs: openArray[float]): NDArray[float] =
  result = newNDArray[float](qs.len)
  for i, q in qs: result.buf[i] = quantile(a, q)

proc iqr*[T](a: NDArray[T]): float = quantile(a, 0.75) - quantile(a, 0.25)

# `quantiles`, `percentile` and `iqr` inherit the propagation from `quantile`
# rather than restating it.

proc nanMedian*[T: SomeFloat](a: NDArray[T]): float =
  ## `median` over the values that are not NaN. Raises when none are, as
  ## `nanMean` does — an array of holes has no middle.
  let xs = withoutNaN(a)
  if xs.len == 0:
    raise newException(ValueError, "nanMedian of an array with no values in it")
  median(initNDArray[T](@[xs.len], xs))

proc nanQuantile*[T: SomeFloat](a: NDArray[T], q: float): float =
  ## `quantile` over the values that are not NaN.
  if q < 0.0 or q > 1.0:
    raise newException(ValueError, "nanQuantile: q must be in 0..1, got " & $q)
  let xs = withoutNaN(a)
  if xs.len == 0:
    raise newException(ValueError, "nanQuantile of an array with no values in it")
  quantile(initNDArray[T](@[xs.len], xs), q)

proc unique*[T](a: NDArray[T]): NDArray[T] =
  ## The distinct values, in sorted order. Distinctness is `==` on the
  ## element type, with one exception: NaN is not equal to itself, so it is
  ## collapsed by hand and appears at most once, at the end. numpy settled on
  ## the same answer — the alternative is a result whose length depends on
  ## how many holes the input happened to have.
  let xs = sortedArray(a).toSeq()
  var out0: seq[T]
  for i, x in xs:
    when T is SomeFloat:
      if x != x:
        if i == 0 or xs[i - 1] == xs[i - 1]: out0.add(x)   # the first NaN only
        continue
    if i == 0 or x != xs[i - 1]: out0.add(x)
  initNDArray[T](@[out0.len], out0)

proc bincount*(a: NDArray[int], minLength = 0): NDArray[int] =
  ## Count occurrences of each non-negative integer, position = value.
  var n = minLength
  for x in a:
    if x < 0: raise newException(ValueError, "bincount: negative value " & $x)
    n = max(n, x + 1)
  result = newNDArray[int](n)
  for x in a: result.buf[x] = result.buf[x] + 1

proc histogram*[T](a: NDArray[T], bins = 10,
                   range: Option[(float, float)] = none((float, float))):
                   (NDArray[int], NDArray[float]) =
  ## Counts and the `bins + 1` edges that produced them. Bins are half-open
  ## `[lo, hi)` except the last, which includes its right edge — otherwise
  ## the maximum value would fall outside every bin.
  ##
  ## The range is the data's own span unless one is given. It is an
  ## `Option` because a tuple has no spare value to mean "not given": the
  ## sentinel this used to carry was `(0.0, 0.0)`, so a caller who really
  ## wanted the interval from 0 to 0 was quietly handed the data's range
  ## instead. Write `range = some((0.0, 6.0))`. A degenerate *derived*
  ## range is widened by half a unit either side, so a constant array still
  ## has a bin to land in, but a degenerate or inverted *given* one raises —
  ## it cannot be anything but a mistake, and it used to return zero counts
  ## and decreasing edges without a word.
  ##
  ## **A hole is not a value and lands in no bin.** NaN is skipped, and the
  ## derived range is taken over the real values, so `sum(counts)` is
  ## `nanCount(a)` rather than `a.size` when the data has holes in it —
  ## `min`/`max` propagate a NaN and would otherwise make every edge NaN and
  ## every count meaningless. Points outside an explicit range are dropped
  ## the same way.
  ##
  ## **An infinity is a value, but it is not a bin edge.** `Inf` and
  ## `NegInf` are kept out of the derived range for the same reason NaN is:
  ## a derived `hi` of `Inf` makes `width` infinite and `lo + 0.0 * Inf` a
  ## NaN, so every edge is unusable and every point falls in bin 0 — the
  ## outcome the NaN filter exists to prevent. They are ordinary values
  ## everywhere else here, so they are not dropped from the data: once the
  ## range is finite they simply lie outside it, like any other point out of
  ## range. A *given* range must be finite for the same reason, and raises
  ## if it is not.
  if bins < 1: raise newException(ValueError, "histogram needs at least one bin")
  if a.size == 0: raise newException(ValueError, "histogram of an empty array")
  var lo, hi: float
  if range.isSome:
    (lo, hi) = range.get
    if classify(lo) in {fcNaN, fcInf, fcNegInf} or
       classify(hi) in {fcNaN, fcInf, fcNegInf}:
      raise newException(ValueError,
        "histogram: range (" & $lo & ", " & $hi & ") must be finite")
    if not (lo < hi):
      raise newException(ValueError,
        "histogram: range lo (" & $lo & ") must be below hi (" & $hi & ")")
  else:
    var first = true
    for x in a:
      if isHole(x): continue
      let v = float(x)
      if classify(v) in {fcInf, fcNegInf}: continue
      if first:
        lo = v
        hi = v
        first = false
      else:
        lo = min(lo, v)
        hi = max(hi, v)
    if first:
      raise newException(ValueError,
        "histogram of an array with no finite values in it: every element " &
        "is NaN or infinite, so there is no range to bin over. " &
        "Pass range = some((lo, hi)).")
    if lo == hi:
      lo -= 0.5
      hi += 0.5
  let width = (hi - lo) / float(bins)
  var counts = newNDArray[int](bins)
  for x in a:
    if isHole(x): continue
    let v = float(x)
    if v < lo or v > hi: continue
    var b = int(floor((v - lo) / width))
    if b >= bins: b = bins - 1        # the right edge belongs to the last bin
    if b < 0: b = 0
    counts.buf[b] = counts.buf[b] + 1
  var edges = newNDArray[float](bins + 1)
  for i in 0 .. bins: edges.buf[i] = lo + float(i) * width
  edges.buf[bins] = hi
  (counts, edges)

proc cov*[T](a, b: NDArray[T], ddof = 1): float =
  ## Sample covariance by default (`ddof = 1`), matching numpy's `cov`.
  if a.size != b.size:
    raise newException(ValueError, "cov: lengths differ")
  if a.size <= ddof:
    raise newException(ValueError, "cov needs more than " & $ddof & " points")
  let ma = mean(a)
  let mb = mean(b)
  var s = 0.0
  let xs = a.toSeq()
  let ys = b.toSeq()
  for i in 0 ..< xs.len: s += (float(xs[i]) - ma) * (float(ys[i]) - mb)
  s / float(xs.len - ddof)

proc corr*[T](a, b: NDArray[T]): float =
  ## Pearson correlation. Undefined when either side is constant, which
  ## raises rather than returning a NaN that would propagate silently.
  let sa = std(a, ddof = 1)
  let sb = std(b, ddof = 1)
  if sa == 0.0 or sb == 0.0:
    raise newException(ValueError, "corr: a constant input has no correlation")
  cov(a, b) / (sa * sb)

proc normalize*[T: SomeFloat](a: NDArray[T]): NDArray[T] =
  ## Rescaled to 0..1 by its own min and max. A constant array maps to zeros
  ## rather than dividing by nothing.
  let lo = min(a)
  let hi = max(a)
  if hi == lo: zeros[T](a.shape) else: (a - lo) / (hi - lo)

proc zscore*[T](a: NDArray[T], ddof = 0): NDArray[float] =
  ## Centred on the mean and scaled by the standard deviation.
  ## Note that `std`/`variance` have both a whole-array and an axis form, so
  ## their second argument must be **named**: `std(a, ddof = 1)`.
  let m = mean(a)
  let s = std(a, ddof = ddof)
  if s == 0.0:
    raise newException(ValueError, "zscore: a constant input has no scale")
  let f = a.astype(float)
  (f - m) / s
