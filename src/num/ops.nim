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

template inplaceZip(dst, src, op: untyped) =
  ## The body every compound assignment shares: broadcast the right-hand
  ## side to the destination, walk the two in lockstep, write through the
  ## view. `op` sees the operands as `x` and `y`, as `zipIt` does.
  ##
  ## The destination is not broadcast — it is where the answer goes, and
  ## stretching it would make one cell the target of many writes. So the
  ## right-hand side must fit `dst` as it stands: `a += rowVector` is fine,
  ## `rowVector += a` is not.
  block:
    # `a += a[0..2]` reads cells the loop has already written, so a source
    # sharing the destination's buffer is materialised first — the same
    # guarantee `setSelect` makes.
    let source = if src.buf == dst.buf: src.copy() else: src
    let rhs = source.broadcastTo(dst.shape)
    for oa, ob in offsets2(dst, rhs):
      let x {.inject.} = dst.buf[oa]
      let y {.inject.} = rhs.buf[ob]
      dst.buf[oa] = op

proc `+=`*[T](a: var NDArray[T], b: NDArray[T]) = inplaceZip(a, b, x + y)
proc `+=`*[T](a: var NDArray[T], s: T) = applyIt(a, it + s)
proc `-=`*[T](a: var NDArray[T], b: NDArray[T]) = inplaceZip(a, b, x - y)
proc `-=`*[T](a: var NDArray[T], s: T) = applyIt(a, it - s)
proc `*=`*[T](a: var NDArray[T], b: NDArray[T]) = inplaceZip(a, b, x * y)
proc `*=`*[T](a: var NDArray[T], s: T) = applyIt(a, it * s)
proc `/=`*[T: SomeFloat](a: var NDArray[T], b: NDArray[T]) = inplaceZip(a, b, x / y)
proc `/=`*[T: SomeFloat](a: var NDArray[T], s: T) = applyIt(a, it / s)
  ## The four compound assignments, each in both forms. `/=` is float-only,
  ## as `/` is: integer division that rounds is `div`, and it would have to
  ## round to write back into an int array.

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

func isHole*[T](x: T): bool {.inline.} =
  ## True only for NaN — **the single elementwise statement of what a hole
  ## is**, so no operation, fold, sort or bin restates `x != x` and none of
  ## them can disagree about it. `isNaN` below is the whole-array reading of
  ## the same question. Non-float element types never reach the branch:
  ## `when` compiles it away, so an int array pays nothing for the check.
  when T is SomeFloat: x != x
  else: false

func minKeepNaN*[T](x, y: T): T {.inline.} =
  ## `min`, with a hole winning. Nim's `min` is written over `<`, which is
  ## false in both directions for NaN, so it silently returns whichever
  ## operand it happened to test first — the same bug the folds had. This
  ## is what `minimum` and `min(axis)` both compare with.
  when T is SomeFloat:
    if isHole(x) or isHole(y): T(NaN) else: system.min(x, y)
  else:
    system.min(x, y)

func maxKeepNaN*[T](x, y: T): T {.inline.} =
  when T is SomeFloat:
    if isHole(x) or isHole(y): T(NaN) else: system.max(x, y)
  else:
    system.max(x, y)

proc isNaN*[T: SomeFloat](a: NDArray[T]): NDArray[bool] = mapIt(a, isHole(it))
proc isFinite*[T: SomeFloat](a: NDArray[T]): NDArray[bool] =
  mapIt(a, not isHole(it) and it != Inf and it != -Inf)

## `maximum`/`minimum` are the *elementwise* pair, as in numpy; `max`/`min`
## in `reductions` are the folds. Both orders of the scalar form are here,
## as they are for every operator above — `maximum(0.0, a)` is how a floor
## reads when the bound is the thing being talked about. All four propagate
## NaN, agreeing with the folds and with numpy's ufuncs.

proc maximum*[T](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, maxKeepNaN(x, y))
proc maximum*[T](a: NDArray[T], s: T): NDArray[T] = maximum(a, scalarArray(s))
proc maximum*[T](s: T, a: NDArray[T]): NDArray[T] = maximum(scalarArray(s), a)
proc minimum*[T](a, b: NDArray[T]): NDArray[T] = zipIt(a, b, minKeepNaN(x, y))
proc minimum*[T](a: NDArray[T], s: T): NDArray[T] = minimum(a, scalarArray(s))
proc minimum*[T](s: T, a: NDArray[T]): NDArray[T] = minimum(scalarArray(s), a)

proc clip*[T](a: NDArray[T], lo, hi: T): NDArray[T] =
  ## Every element brought inside `lo .. hi`. An inverted bound is a caller's
  ## mistake, not a request: `clip(a, 3, 1)` used to give whichever end each
  ## element was compared against first, which is neither bound and no
  ## intelligible answer.
  if lo > hi:
    raise newException(ValueError,
      "clip: lo (" & $lo & ") is above hi (" & $hi & ")")
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
