## Elementwise operations — numpy's ufuncs.
##
## Everything here is built from two templates, `mapIt` (one array) and
## `zipIt` (two, broadcast against each other), so broadcasting, the result
## element type, and the traversal are each stated once. An operator is then
## a one-liner, and adding one costs a line rather than a loop.
##
## Element-wise equality is `eq`/`neq`, **not** `==`/`!=`: `==` already means
## whole-array equality and returns a `bool`, and Nim cannot overload on the
## return type. This matches racoon's `Column`, deliberately.

import std/math
import ./core
import ./shape
import ./creation

# ------------------------------------------------------------ the two -----

template mapIt*(a: untyped, op: untyped): untyped =
  ## `a` with `op` applied to each element, which `op` sees as `it`. The
  ## result's element type is whatever `op` returns, so
  ## `x.mapIt(it > 0)` gives an `NDArray[bool]` with no extra machinery.
  block:
    let src = a
    type R = typeof((
      block:
        var it {.inject.}: typeof(src.buf[0])
        op))
    var res = newNDArray[R](src.shape)
    var i = 0
    for elem in src:
      let it {.inject.} = elem
      res.buf[i] = op
      i.inc
    res

template zipIt*(a, b: untyped, op: untyped): untyped =
  ## `op` over two arrays broadcast to their common shape, seeing the
  ## operands as `x` and `y`.
  block:
    let (xs, ys) = broadcast2(a, b)
    type R = typeof((
      block:
        var x {.inject.}: typeof(xs.buf[0])
        var y {.inject.}: typeof(ys.buf[0])
        op))
    var res = newNDArray[R](xs.shape)
    var i = 0
    for oa, ob in offsets2(xs, ys):
      let x {.inject.} = xs.buf[oa]
      let y {.inject.} = ys.buf[ob]
      res.buf[i] = op
      i.inc
    res

template applyIt*(a: var untyped, op: untyped) =
  ## In-place `mapIt`: writes through the view, so `a` may be a slice of a
  ## larger array and only that window changes.
  for off in a.offsets:
    let it {.inject.} = a.buf[off]
    a.buf[off] = op

# --------------------------------------------------------- arithmetic -----

proc scalarArray*[T](s: T): NDArray[T] =
  ## A 0-d array holding `s`. Scalar operands go through this rather than
  ## through their own loop: a 0-d array broadcasts to any shape with strides
  ## of 0, so `a + 1` and `a + b` are the same code path.
  initNDArray[T](@[], @[s])

template binOp(name, body: untyped) =
  proc name*[T](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, body)
  proc name*[T](a: NDArray[T], s: T): NDArray[T] = name(a, scalarArray(s))
  proc name*[T](s: T, a: NDArray[T]): NDArray[T] = name(scalarArray(s), a)

binOp(`+`, x + y)
binOp(`-`, x - y)
binOp(`*`, x * y)

proc `-`*[T](a: NDArray[T]): NDArray[T] = mapIt(a, -it)

proc `/`*[T: SomeFloat](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, x / y)
proc `/`*[T: SomeFloat](a: NDArray[T], s: T): NDArray[T] = mapIt(a, it / s)
proc `/`*[T: SomeFloat](s: T, a: NDArray[T]): NDArray[T] = mapIt(a, s / it)

proc `/`*(a, b: NDArray[int]): NDArray[float] = zipIt(a, b, x.float / y.float)
proc `/`*(a: NDArray[int], s: int): NDArray[float] = mapIt(a, it.float / s.float)
proc `/`*(s: int, a: NDArray[int]): NDArray[float] = mapIt(a, s.float / it.float)
  ## Integer division gives floats, as numpy's `/` does; `div` is the
  ## truncating one. This is the same rule racoon's `/` follows.

proc `div`*(a, b: NDArray[int]): NDArray[int] = zipIt(a, b, x div y)
proc `div`*(a: NDArray[int], s: int): NDArray[int] = mapIt(a, it div s)
proc `mod`*(a, b: NDArray[int]): NDArray[int] = zipIt(a, b, x mod y)
proc `mod`*(a: NDArray[int], s: int): NDArray[int] = mapIt(a, it mod s)

proc `+=`*[T](a: var NDArray[T], b: NDArray[T]) =
  let src = if b.buf == a.buf: b.copy() else: b   # see `setSelect`: views alias
  let rhs = src.broadcastTo(a.shape)
  for oa, ob in offsets2(a, rhs): a.buf[oa] = a.buf[oa] + rhs.buf[ob]
proc `+=`*[T](a: var NDArray[T], s: T) = applyIt(a, it + s)
proc `-=`*[T](a: var NDArray[T], b: NDArray[T]) =
  let src = if b.buf == a.buf: b.copy() else: b   # see `setSelect`: views alias
  let rhs = src.broadcastTo(a.shape)
  for oa, ob in offsets2(a, rhs): a.buf[oa] = a.buf[oa] - rhs.buf[ob]
proc `-=`*[T](a: var NDArray[T], s: T) = applyIt(a, it - s)
proc `*=`*[T](a: var NDArray[T], s: T) = applyIt(a, it * s)
proc `/=`*[T: SomeFloat](a: var NDArray[T], s: T) = applyIt(a, it / s)

# -------------------------------------------------------- comparisons -----

template cmpOp(name, body: untyped) =
  proc name*[T](a, b: NDArray[T]): NDArray[bool] = zipIt(a, b, body)
  proc name*[T](a: NDArray[T], s: T): NDArray[bool] = name(a, scalarArray(s))
  proc name*[T](s: T, a: NDArray[T]): NDArray[bool] = name(scalarArray(s), a)

cmpOp(`<`, x < y)
cmpOp(`<=`, x <= y)
cmpOp(`>`, x > y)
cmpOp(`>=`, x >= y)
cmpOp(eq, x == y)
cmpOp(neq, x != y)

proc `and`*(a, b: NDArray[bool]): NDArray[bool] = zipIt(a, b, x and y)
proc `or`*(a, b: NDArray[bool]): NDArray[bool] = zipIt(a, b, x or y)
proc `xor`*(a, b: NDArray[bool]): NDArray[bool] = zipIt(a, b, x xor y)
proc `not`*(a: NDArray[bool]): NDArray[bool] = mapIt(a, not it)

proc allclose*[T: SomeFloat](a, b: NDArray[T], rtol = 1e-5, atol = 1e-8): bool =
  ## Elementwise closeness with numpy's tolerance rule
  ## (`|a-b| <= atol + rtol*|b|`), for the comparisons exact equality is the
  ## wrong test for. NaN is never close to anything.
  let (xs, ys) = broadcast2(a, b)
  for oa, ob in offsets2(xs, ys):
    let u = xs.buf[oa]
    let v = ys.buf[ob]
    if u != u or v != v: return false
    if abs(u - v) > atol + rtol * abs(v): return false
  true

# --------------------------------------------------------------- math -----

template mathFn(name: untyped) =
  proc name*[T: SomeFloat](a: NDArray[T]): NDArray[T] = mapIt(a, math.name(it))

mathFn(sqrt)
mathFn(exp)
mathFn(ln)
mathFn(log10)
mathFn(log2)
mathFn(sin)
mathFn(cos)
mathFn(tan)
mathFn(arcsin)
mathFn(arccos)
mathFn(arctan)
mathFn(sinh)
mathFn(cosh)
mathFn(tanh)
mathFn(floor)
mathFn(ceil)
mathFn(round)
mathFn(trunc)
mathFn(degToRad)
mathFn(radToDeg)

proc abs*[T](a: NDArray[T]): NDArray[T] = mapIt(a, abs(it))
proc square*[T](a: NDArray[T]): NDArray[T] = mapIt(a, it * it)
proc sign*[T](a: NDArray[T]): NDArray[T] =
  mapIt(a, (if it > T(0): T(1) elif it < T(0): T(-1) else: T(0)))
proc reciprocal*[T: SomeFloat](a: NDArray[T]): NDArray[T] = mapIt(a, T(1) / it)

proc pow*[T: SomeFloat](a: NDArray[T], p: T): NDArray[T] = mapIt(a, math.pow(it, p))
proc pow*[T: SomeFloat](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, math.pow(x, y))
proc `^`*[T: SomeFloat](a: NDArray[T], p: T): NDArray[T] = pow(a, p)

proc isNaN*[T: SomeFloat](a: NDArray[T]): NDArray[bool] = mapIt(a, it != it)
proc isFinite*[T: SomeFloat](a: NDArray[T]): NDArray[bool] =
  mapIt(a, it == it and it != Inf and it != -Inf)

proc maximum*[T](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, max(x, y))
proc maximum*[T](a: NDArray[T], s: T): NDArray[T] = mapIt(a, max(it, s))
proc minimum*[T](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, min(x, y))
proc minimum*[T](a: NDArray[T], s: T): NDArray[T] = mapIt(a, min(it, s))
  ## `maximum`/`minimum` are the *elementwise* pair, as in numpy; `max`/`min`
  ## in `reductions` are the folds.

proc clip*[T](a: NDArray[T], lo, hi: T): NDArray[T] =
  mapIt(a, (if it < lo: lo elif it > hi: hi else: it))

# ------------------------------------------------------------ selection ----

proc where*[T](cond: NDArray[bool], a, b: NDArray[T]): NDArray[T] =
  ## Elementwise choice, all three operands broadcast together. The condition
  ## is copied contiguous first: `offsets2` walks two arrays and a third
  ## odometer here would be a second statement of the same traversal.
  let shape = broadcastShapes(broadcastShapes(cond.shape, a.shape), b.shape)
  let c = cond.broadcastTo(shape).copy()
  let x = a.broadcastTo(shape)
  let y = b.broadcastTo(shape)
  result = newNDArray[T](shape)
  var i = 0
  for ox, oy in offsets2(x, y):
    result.buf[i] = (if c.buf[i]: x.buf[ox] else: y.buf[oy])
    i.inc

proc where*[T](cond: NDArray[bool], a: NDArray[T], s: T): NDArray[T] =
  where(cond, a, full(a.shape, s))
proc where*[T](cond: NDArray[bool], s: T, b: NDArray[T]): NDArray[T] =
  where(cond, full(b.shape, s), b)
