## Making arrays: from literals, from ranges, and from nothing.

import std/[math, sequtils]
import ./core
import ./shape
import ./index

# ------------------------------------------------------ from Nim literals ---

proc toNDArray*[T](xs: openArray[T]): NDArray[T] =
  ## 1-d from a flat seq/array.
  initNDArray[T](@[xs.len], @xs)

proc toNDArray*[T](xs: openArray[seq[T]]): NDArray[T] =
  ## 2-d from a seq of rows, which must all be the same length.
  let rows = xs.len
  let cols = if rows == 0: 0 else: xs[0].len
  var data = newSeqOfCap[T](rows * cols)
  for r in xs:
    if r.len != cols:
      raise newException(ValueError,
        "ragged rows: " & $cols & " and " & $r.len)
    data.add(r)
  initNDArray[T](@[rows, cols], data)

proc toNDArray*[T](xs: openArray[seq[seq[T]]]): NDArray[T] =
  ## 3-d from nested seqs, all planes and rows the same size.
  let n0 = xs.len
  let n1 = if n0 == 0: 0 else: xs[0].len
  let n2 = if n1 == 0: 0 else: xs[0][0].len
  var data = newSeqOfCap[T](n0 * n1 * n2)
  for plane in xs:
    if plane.len != n1:
      raise newException(ValueError, "ragged planes")
    for r in plane:
      if r.len != n2:
        raise newException(ValueError, "ragged rows")
      data.add(r)
  initNDArray[T](@[n0, n1, n2], data)

proc arr*[T](xs: openArray[T]): NDArray[T] = toNDArray(xs)
proc arr*[T](xs: openArray[seq[T]]): NDArray[T] = toNDArray(xs)
proc arr*[T](xs: openArray[seq[seq[T]]]): NDArray[T] = toNDArray(xs)
  ## Short alias for `toNDArray`, for the places a literal reads better
  ## without the noise: `arr(@[1, 2, 3])`.

# ------------------------------------------------------------- filled ------

proc zeros*[T](shape: varargs[int]): NDArray[T] =
  ## The element type is always explicit — `zeros[float](3, 3)`. Nim cannot
  ## give a generic parameter a default *and* infer it from nothing, and an
  ## overload pair to fake one is ambiguous, so rather than have `zeros`
  ## work differently from `ones` and `eye`, all of them ask.
  newNDArray[T](shape)

proc ones*[T](shape: varargs[int]): NDArray[T] =
  result = newNDArray[T](shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = T(1)

proc full*[T](shape: openArray[int], val: T): NDArray[T] =
  result = newNDArray[T](shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = val

proc empty*[T](shape: varargs[int]): NDArray[T] =
  ## Allocated but unspecified — in Nim that means zeroed, so this is
  ## `zeros` under another name, kept for numpy familiarity.
  newNDArray[T](shape)

proc zerosLike*[T](a: NDArray[T]): NDArray[T] = newNDArray[T](a.shape)
proc onesLike*[T](a: NDArray[T]): NDArray[T] = ones[T](a.shape)
proc fullLike*[T](a: NDArray[T], val: T): NDArray[T] = full(a.shape, val)

# -------------------------------------------------------------- ranges -----

proc arange*[T](start, stop: T, step: T = T(1)): NDArray[T] =
  ## Half-open `[start, stop)`, like Python's `range` and numpy's `arange`.
  ## A zero step never terminates and is rejected.
  if step == T(0):
    raise newException(ValueError, "arange: step must be non-zero")
  var n = 0
  when T is SomeFloat:
    n = max(0, int(ceil((stop - start) / step)))
  else:
    let span = stop - start
    n = if (span > 0) == (step > 0): max(0, (span + step - (if step > 0: 1 else: -1)) div step)
        else: 0
  # each point is computed from `start`, never accumulated: adding `step` to
  # a running total drifts, and `arange(0.0, 1.0, 0.1)` ended at
  # 0.8999999999999999. This is the care `linspace` takes over its endpoint.
  result = newNDArray[T](n)
  for i in 0 ..< n:
    result.buf[i] = start + T(i) * step

proc arange*[T](stop: T): NDArray[T] = arange(T(0), stop, T(1))
  ## `arange` infers its type from its arguments: `arange(6)` is int,
  ## `arange(0.0, 1.0, 0.25)` is float.

proc linspace*(start, stop: float, num = 50, endpoint = true): NDArray[float] =
  ## `num` evenly spaced points. With `endpoint`, `stop` is the last one and
  ## the step is `(stop-start)/(num-1)`; without, it is `/num` and `stop` is
  ## excluded. The last point is written from `stop` directly rather than
  ## accumulated, so it is exactly `stop`: `linspace(0, 1, n)[^1].item` is
  ## `1.0`. (`^k` is a selection, not an integer index, so `[^1]` is a 0-d
  ## array — `item` unwraps it.)
  if num < 0:
    raise newException(ValueError, "linspace: num must be non-negative")
  result = newNDArray[float](num)
  if num == 0: return
  if num == 1:
    result.buf[0] = start
    return
  let div0 = if endpoint: float(num - 1) else: float(num)
  let step = (stop - start) / div0
  for i in 0 ..< num:
    result.buf[i] = start + float(i) * step
  if endpoint: result.buf[num - 1] = stop

proc logspace*(start, stop: float, num = 50, endpoint = true,
               base = 10.0): NDArray[float] =
  ## `num` points evenly spaced on a log scale, i.e. `base^linspace(...)`.
  ## The exponents are what is evenly spaced, so `endpoint` means what it
  ## means for `linspace` and is passed straight through — it was silently
  ## dropped before, which made `endpoint = false` unaskable here.
  ## The parameter order is numpy's: `endpoint` before `base`.
  result = linspace(start, stop, num, endpoint)
  for i in 0 ..< result.buf[].len:
    result.buf[i] = pow(base, result.buf[i])

# ------------------------------------------------------------ matrices -----

proc eye*[T](n: int, m = -1, k = 0): NDArray[T] =
  ## `n` x `m` (square by default) with ones on the `k`-th diagonal.
  let cols = if m < 0: n else: m
  result = newNDArray[T](n, cols)
  for i in 0 ..< n:
    let j = i + k
    if j >= 0 and j < cols:
      result[i, j] = T(1)

proc identity*[T](n: int): NDArray[T] = eye[T](n)

proc meshgrid*[T](x, y: NDArray[T]): (NDArray[T], NDArray[T]) =
  ## Coordinate matrices from two 1-d vectors: `xx` varies along the columns
  ## and `yy` along the rows, the `indexing="xy"` default. Both are broadcast
  ## views, so the pair costs nothing until it is written to.
  if x.ndim != 1 or y.ndim != 1:
    raise newException(ValueError, "meshgrid takes 1-d arrays")
  let shape = @[y.size, x.size]
  (x.reshape(1, x.size).broadcastTo(shape), y.reshape(y.size, 1).broadcastTo(shape))
