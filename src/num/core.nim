## The array itself: shape, strides, and the buffer they address.
##
## `NDArray[T]` is a *view* onto a shared buffer — `shape`/`strides`/`offset`
## say which elements of `buf` belong to it and in what order. Reshaping,
## transposing, slicing and broadcasting all produce a new view over the same
## buffer, so they are O(1) and write-through, exactly as numpy's are. `copy`
## is the one thing that allocates.
##
## Strides are counted in **elements**, not bytes: Nim indexes a `seq[T]`, so
## there is no reason to carry an element size around. A stride of 0 is legal
## and is how broadcasting repeats an axis without copying it.

import std/[math, strutils, sequtils]

type
  NDArray*[T] = object
    ## An n-dimensional array. A 0-d array (`shape.len == 0`) is a scalar and
    ## holds exactly one element; that is what an axis-reduction of a 1-d
    ## array collapses to before `item` unwraps it.
    shape*: seq[int]
    strides*: seq[int]
    offset*: int
    buf*: ref seq[T]

# ------------------------------------------------------------- geometry ----

func prod*(xs: openArray[int]): int =
  ## Element count implied by a shape. The empty shape is a scalar, so this
  ## is 1 rather than 0 — the empty product.
  result = 1
  for x in xs: result *= x

func contiguousStrides*(shape: openArray[int]): seq[int] =
  ## Row-major (C-order) strides for `shape`. The single statement of what
  ## "contiguous" means; every constructor and `reshape` goes through it.
  result = newSeq[int](shape.len)
  var acc = 1
  for i in countdown(shape.high, 0):
    result[i] = acc
    acc *= shape[i]

func ndim*[T](a: NDArray[T]): int = a.shape.len
func size*[T](a: NDArray[T]): int = prod(a.shape)
func len*[T](a: NDArray[T]): int =
  ## Length along the first axis, following numpy — *not* the element count,
  ## which is `size`. A 0-d array has no first axis and so has length 0.
  if a.shape.len == 0: 0 else: a.shape[0]

func isContiguous*[T](a: NDArray[T]): bool =
  ## True when the elements sit in `buf` in logical order starting at
  ## `offset`, which is what lets bulk paths index the payload directly.
  ## Axes of length 1 carry no information about order, so their stride is
  ## not checked — a stride of anything (0 included) is contiguous there.
  var acc = 1
  for i in countdown(a.shape.high, 0):
    if a.shape[i] != 1:
      if a.strides[i] != acc: return false
    acc *= a.shape[i]
  true

# --------------------------------------------------------- construction ----

func checkShape(shape: openArray[int]) =
  for s in shape:
    if s < 0:
      raise newException(ValueError, "negative dimension in shape " & $(@shape))

proc newNDArray*[T](shape: varargs[int]): NDArray[T] =
  ## A fresh array of `T.default` (zeros for the numeric types).
  checkShape(shape)
  result = NDArray[T](shape: @shape, strides: contiguousStrides(shape),
                      offset: 0, buf: new(seq[T]))
  result.buf[] = newSeq[T](prod(shape))

proc initNDArray*[T](shape: openArray[int], data: sink seq[T]): NDArray[T] =
  ## Wrap `data` as an array of `shape`, taking ownership rather than copying.
  checkShape(shape)
  if prod(shape) != data.len:
    raise newException(ValueError,
      "shape " & $(@shape) & " needs " & $prod(shape) & " elements, got " & $data.len)
  result = NDArray[T](shape: @shape, strides: contiguousStrides(shape),
                      offset: 0, buf: new(seq[T]))
  result.buf[] = data

proc view*[T](a: NDArray[T], shape, strides: sink seq[int], offset: int): NDArray[T] =
  ## A new window onto `a`'s buffer. The primitive every O(1) reshaping
  ## operation is expressed in; nothing outside this module builds an
  ## `NDArray` literal.
  NDArray[T](shape: shape, strides: strides, offset: offset, buf: a.buf)

# -------------------------------------------------------------- walking ----

iterator offsets*[T](a: NDArray[T]): int =
  ## Buffer positions of `a`'s elements in logical (row-major) order. The one
  ## traversal that copes with any strides — negative, zero, or overlapping —
  ## so every generic operation is written over it rather than over `buf`.
  if a.size > 0:
    if a.shape.len == 0:
      yield a.offset
    else:
      var idx = newSeq[int](a.shape.len)
      var off = a.offset
      let n = a.size
      for _ in 0 ..< n:
        yield off
        # odometer: advance the last axis, carrying into the ones before it
        for ax in countdown(a.shape.high, 0):
          idx[ax] += 1
          off += a.strides[ax]
          if idx[ax] < a.shape[ax]: break
          off -= a.strides[ax] * a.shape[ax]
          idx[ax] = 0

iterator items*[T](a: NDArray[T]): T =
  for off in a.offsets: yield a.buf[off]

iterator pairs*[T](a: NDArray[T]): (seq[int], T) =
  ## Multi-index and value. The index seq is reused between iterations, so
  ## copy it if you keep it.
  var idx = newSeq[int](a.shape.len)
  for off in a.offsets:
    yield (idx, a.buf[off])
    for ax in countdown(a.shape.high, 0):
      idx[ax] += 1
      if idx[ax] < a.shape[ax]: break
      idx[ax] = 0

func toSeq*[T](a: NDArray[T]): seq[T] =
  ## The elements in logical order, as a flat `seq`. Always a copy.
  result = newSeqOfCap[T](a.size)
  for off in a.offsets: result.add(a.buf[off])

proc copy*[T](a: NDArray[T]): NDArray[T] =
  ## A contiguous array with the same shape and values, sharing nothing.
  initNDArray[T](a.shape, a.toSeq())

proc asContiguous*[T](a: NDArray[T]): NDArray[T] =
  ## `a` itself when it is already contiguous, a copy otherwise. Use it
  ## before a fast path that indexes `buf` directly.
  if a.isContiguous and a.offset == 0 and a.buf[].len == a.size: a else: a.copy()

# ------------------------------------------------------------- indexing ----

func flatOffset*[T](a: NDArray[T], idx: openArray[int]): int =
  ## Buffer position of one element, resolving negative indices from the end
  ## the way numpy does. The single bounds check for scalar access.
  if idx.len != a.shape.len:
    raise newException(ValueError,
      $idx.len & " indices for a " & $a.shape.len & "-d array: an all-integer " &
      "index must name every axis. Use `All` for the rest (a[1, All]) to get a view.")
  result = a.offset
  for ax, i in idx:
    let j = if i < 0: i + a.shape[ax] else: i
    if j < 0 or j >= a.shape[ax]:
      raise newException(ValueError,
        "index " & $i & " out of range for axis " & $ax & " of length " & $a.shape[ax])
    result += j * a.strides[ax]

func elemAt*[T](a: NDArray[T], idx: varargs[int]): T =
  ## One element. Users reach this through `[]` (see `index.nim`), which
  ## picks between element access and slicing; the two cannot be overloads of
  ## each other because an `int` converts to a `Sel` and every all-int call
  ## would then be ambiguous.
  a.buf[a.flatOffset(idx)]

proc setElem*[T](a: var NDArray[T], idx: openArray[int], val: T) =
  a.buf[a.flatOffset(idx)] = val

func item*[T](a: NDArray[T]): T =
  ## The single element of a one-element array — what a full reduction
  ## returns before it is used as a scalar.
  if a.size != 1:
    raise newException(ValueError,
      "item() needs exactly one element, array has " & $a.size)
  for x in a: return x

proc fill*[T](a: var NDArray[T], val: T) =
  ## Write `val` into every element *of this view*, base array included.
  for off in a.offsets: a.buf[off] = val

# --------------------------------------------------------------- equality --

iterator offsets2*[T, U](a: NDArray[T], b: NDArray[U]): (int, int) =
  ## Two arrays of the same shape walked in lockstep. Every binary operation
  ## on already-broadcast operands is written over this, so neither side has
  ## to be contiguous and neither is materialised.
  if a.shape != b.shape:
    raise newException(ValueError,
      "shape mismatch: " & $a.shape & " vs " & $b.shape)
  if a.size > 0:
    if a.shape.len == 0:
      yield (a.offset, b.offset)
    else:
      var idx = newSeq[int](a.shape.len)
      var oa = a.offset
      var ob = b.offset
      for _ in 0 ..< a.size:
        yield (oa, ob)
        for ax in countdown(a.shape.high, 0):
          idx[ax] += 1
          oa += a.strides[ax]
          ob += b.strides[ax]
          if idx[ax] < a.shape[ax]: break
          oa -= a.strides[ax] * a.shape[ax]
          ob -= b.strides[ax] * a.shape[ax]
          idx[ax] = 0

func `==`*[T](a, b: NDArray[T]): bool =
  ## Structural: same shape and same values in logical order. Two arrays can
  ## be equal with different strides, and an array equals its own copy.
  if a.shape != b.shape: return false
  for oa, ob in offsets2(a, b):
    if a.buf[oa] != b.buf[ob]: return false
  true

# --------------------------------------------------------------- printing --

func trimZeros(s: string): string =
  ## `1.00000` -> `1.0`, `0.500000` -> `0.5`, mantissa only for `1.00000e-12`.
  let e = s.find('e')
  if e >= 0: return trimZeros(s[0 ..< e]) & s[e .. ^1]
  if '.' notin s: return s
  result = s.strip(leading = false, chars = {'0'})
  if result.endsWith('.'): result.add('0')

func formatElem[T](x: T): string =
  ## Floats print to six significant digits, as numpy's `repr` does — a
  ## column of `0.048621663832631515` is unreadable and the full value is
  ## still one `toSeq` away. `$` on a single element is Nim's own and is
  ## unaffected.
  when T is SomeFloat:
    if x != x: "nan"
    elif x == Inf: "inf"
    elif x == -Inf: "-inf"
    else: trimZeros(formatFloat(x, ffDefault, 6))
  elif T is string: "\"" & x & "\""
  else: $x

const
  printThreshold = 1000
    ## Above this many elements `$` prints a summary rather than the whole
    ## array. numpy's default, and for the same reason: a screen of numbers
    ## nobody reads is worse than a shape and a few corners. `toSeq` is
    ## always there when every value is wanted.
  printEdgeItems = 3
    ## How many entries survive at each end of a truncated axis.

iterator shown(n, edge: int): int =
  ## The positions printed along an axis of length `n`: all of them, or the
  ## first and last `edge` with `-1` marking the gap between. The one
  ## statement of what truncation means — the width scan and the render both
  ## walk this, so they cannot disagree about which elements appear.
  if edge <= 0 or n <= 2 * edge:
    for i in 0 ..< n: yield i
  else:
    for i in 0 ..< edge: yield i
    yield -1
    for i in n - edge ..< n: yield i

proc scanWidth[T](a: NDArray[T], idx: var seq[int], ax, edge: int,
                  width: var int) =
  ## The widest cell among the ones that will actually be printed. Measuring
  ## every element instead would walk the whole array for the summary too.
  for i in shown(a.shape[ax], edge):
    if i < 0: continue
    idx[ax] = i
    if ax == a.shape.high:
      width = max(width, formatElem(a.buf[a.flatOffset(idx)]).len)
    else:
      scanWidth(a, idx, ax + 1, edge, width)

proc renderRec[T](a: NDArray[T], idx: var seq[int], ax, edge, width: int,
                  res: var string) =
  let indent = " ".repeat(7 + ax)   # "array(" plus one bracket per level
  let last = ax == a.shape.high
  res.add("[")
  var first = true
  for i in shown(a.shape[ax], edge):
    if not first:
      res.add(if last: ", " else: ",\n" & indent)
    first = false
    if i < 0:
      res.add(if last: align("...", width) else: "...")
    elif last:
      idx[ax] = i
      res.add(align(formatElem(a.buf[a.flatOffset(idx)]), width))
    else:
      idx[ax] = i
      renderRec(a, idx, ax + 1, edge, width, res)
  res.add("]")

proc `$`*[T](a: NDArray[T]): string =
  ## numpy-ish: nested brackets, one row per line, cells right-aligned to a
  ## common width so columns line up. Past `printThreshold` elements only the
  ## corners are shown, with `...` standing in for the rest.
  if a.shape.len == 0:
    return "array(" & formatElem(a.item) & ")"
  if a.size == 0:
    # the shape is spelled as `describe` spells it: `$seq` would put Nim's
    # `@[0, 3]` in front of a user who never mentioned a seq
    return "array([], shape=(" & a.shape.join(", ") & "))"
  let edge = if a.size > printThreshold: printEdgeItems else: 0
  var idx = newSeq[int](a.shape.len)
  var width = 0
  scanWidth(a, idx, 0, edge, width)
  result = "array("
  renderRec(a, idx, 0, edge, width, result)
  result.add(")")
