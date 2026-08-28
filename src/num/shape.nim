## Changing an array's geometry without touching its values.
##
## Everything here returns a **view** where it can (`reshape` of a contiguous
## array, `transpose`, `squeeze`, `broadcastTo`) and copies only when the
## requested geometry cannot be expressed as strides over the existing buffer.
## `reshape` is the sole exception that may or may not copy; the doc comment
## says so at the call site.

import std/[algorithm, sequtils]
import ./core

# -------------------------------------------------------------- reshape ----

func resolveShape(shape: openArray[int], size: int): seq[int] =
  ## Fill in a single `-1` from the element count, numpy-style. More than one
  ## `-1` is ambiguous and is rejected rather than guessed.
  result = @shape
  var wild = -1
  var known = 1
  for i, s in result:
    if s == -1:
      if wild >= 0:
        raise newException(ValueError, "only one dimension may be -1")
      wild = i
    elif s < 0:
      raise newException(ValueError, "negative dimension in shape " & $(@shape))
    else:
      known *= s
  if wild >= 0:
    if known == 0 or size mod known != 0:
      raise newException(ValueError,
        "cannot reshape " & $size & " elements into " & $(@shape))
    result[wild] = size div known
  elif known != size:
    raise newException(ValueError,
      "cannot reshape " & $size & " elements into " & $(@shape))

proc reshape*[T](a: NDArray[T], shape: varargs[int]): NDArray[T] =
  ## Same elements, new shape, in row-major order. One dimension may be `-1`
  ## and is inferred. A view when `a` is contiguous, a copy when it is not —
  ## the layout of a transposed or strided array is not reachable by strides
  ## alone under a different shape.
  let want = resolveShape(shape, a.size)
  let src = if a.isContiguous: a else: a.copy()
  view(src, want, contiguousStrides(want), src.offset)

proc ravel*[T](a: NDArray[T]): NDArray[T] =
  ## Flattened to 1-d; a view when possible, like `reshape`.
  a.reshape(a.size)

proc flatten*[T](a: NDArray[T]): NDArray[T] =
  ## Flattened to 1-d, always a copy.
  initNDArray[T](@[a.size], a.toSeq())

# ------------------------------------------------------------ transpose ----

func normAxis*(ax, ndim: int): int =
  ## Resolve a possibly-negative axis. The one place `-1` becomes "the last
  ## axis", so no operation can disagree about it.
  result = if ax < 0: ax + ndim else: ax
  if result < 0 or result >= ndim:
    raise newException(ValueError,
      "axis " & $ax & " out of range for a " & $ndim & "-d array")

proc transpose*[T](a: NDArray[T], axes: varargs[int]): NDArray[T] =
  ## Permute the axes; with no arguments, reverse them (matrix transpose for
  ## 2-d). Always a view — a permutation of the axes is a permutation of the
  ## strides.
  var perm: seq[int]
  if axes.len == 0:
    perm = newSeq[int](a.ndim)
    for i in 0 ..< a.ndim: perm[i] = a.ndim - 1 - i
  else:
    if axes.len != a.ndim:
      raise newException(ValueError,
        "axis permutation of length " & $axes.len & " for a " & $a.ndim & "-d array")
    perm = newSeq[int](axes.len)
    for i, ax in axes: perm[i] = normAxis(ax, a.ndim)
    if sorted(perm) != toSeq(0 ..< a.ndim):
      raise newException(ValueError, "axes " & $(@axes) & " is not a permutation")
  var shape = newSeq[int](perm.len)
  var strides = newSeq[int](perm.len)
  for i, ax in perm:
    shape[i] = a.shape[ax]
    strides[i] = a.strides[ax]
  view(a, shape, strides, a.offset)

proc t*[T](a: NDArray[T]): NDArray[T] =
  ## `a.T` in numpy: the full axis reversal.
  transpose(a)

proc swapAxes*[T](a: NDArray[T], ax1, ax2: int): NDArray[T] =
  var perm = toSeq(0 ..< a.ndim)
  let i = normAxis(ax1, a.ndim)
  let j = normAxis(ax2, a.ndim)
  swap(perm[i], perm[j])
  transpose(a, perm)

proc moveAxis*[T](a: NDArray[T], src, dst: int): NDArray[T] =
  ## Move one axis to a new position, sliding the others along.
  let s = normAxis(src, a.ndim)
  let d = normAxis(dst, a.ndim)
  var perm = toSeq(0 ..< a.ndim)
  perm.delete(s)
  perm.insert(s, d)
  transpose(a, perm)

# ---------------------------------------------------------------- flip ----

proc flip*[T](a: NDArray[T], axis: int): NDArray[T] =
  ## Reverse the order of one axis. A view: the axis keeps its length and
  ## its stride changes sign, with the offset moved to what was the last
  ## element. Nothing is copied, so this aliases `a` like every other view
  ## here: bind it to a `var` and write through it and `a` changes with it.
  let ax = normAxis(axis, a.ndim)
  var strides = a.strides
  strides[ax] = -a.strides[ax]
  # an empty axis has no last element to start from; the offset is never
  # dereferenced in that case, but it should still be a position in the buffer
  let back = max(a.shape[ax] - 1, 0)
  view(a, a.shape, strides, a.offset + back * a.strides[ax])

proc flip*[T](a: NDArray[T]): NDArray[T] =
  ## Reverse every axis at once, as numpy's `flip` with no axis does. For a
  ## 2-d array that is a 180-degree rotation, not a transpose.
  result = a
  for ax in 0 ..< a.ndim: result = flip(result, ax)

# --------------------------------------------------- adding / dropping 1 ----

proc expandDims*[T](a: NDArray[T], axis: int): NDArray[T] =
  ## Insert an axis of length 1. Its stride is irrelevant (the axis has one
  ## position), so it takes the stride of whatever follows it.
  let ax = normAxis(axis, a.ndim + 1)
  var shape = a.shape
  var strides = a.strides
  let st = if ax < a.ndim: a.strides[ax] * a.shape[ax] else: 1
  shape.insert(1, ax)
  strides.insert(st, ax)
  view(a, shape, strides, a.offset)

proc squeeze*[T](a: NDArray[T]): NDArray[T] =
  ## Drop every axis of length 1.
  var shape, strides: seq[int]
  for i in 0 ..< a.ndim:
    if a.shape[i] != 1:
      shape.add(a.shape[i])
      strides.add(a.strides[i])
  view(a, shape, strides, a.offset)

proc squeeze*[T](a: NDArray[T], axis: int): NDArray[T] =
  ## Drop one axis, which must have length 1.
  let ax = normAxis(axis, a.ndim)
  if a.shape[ax] != 1:
    raise newException(ValueError,
      "cannot squeeze axis " & $axis & " of length " & $a.shape[ax])
  var shape = a.shape
  var strides = a.strides
  shape.delete(ax)
  strides.delete(ax)
  view(a, shape, strides, a.offset)

# ------------------------------------------------------------ broadcast ----

func broadcastShapes*(a, b: openArray[int]): seq[int] =
  ## numpy's rule, stated once: align the shapes at their **trailing** axes,
  ## and each pair must be equal or one of them 1, the 1 being the side that
  ## stretches. Missing leading axes count as 1.
  let n = max(a.len, b.len)
  result = newSeq[int](n)
  for k in 0 ..< n:
    let da = if k < n - a.len: 1 else: a[k - (n - a.len)]
    let db = if k < n - b.len: 1 else: b[k - (n - b.len)]
    if da == db: result[k] = da
    elif da == 1: result[k] = db
    elif db == 1: result[k] = da
    else:
      raise newException(ValueError,
        "shapes " & $(@a) & " and " & $(@b) & " are not broadcastable")

proc broadcastTo*[T](a: NDArray[T], shape: openArray[int]): NDArray[T] =
  ## A view of `a` with the given shape, repeating stretched axes by giving
  ## them a **stride of 0** rather than copying. The result therefore aliases
  ## `a` many times over; do not write through it.
  if shape.len < a.ndim:
    raise newException(ValueError,
      "cannot broadcast " & $a.shape & " to the shorter " & $(@shape))
  let pad = shape.len - a.ndim
  var strides = newSeq[int](shape.len)
  for k in 0 ..< shape.len:
    if k < pad:
      strides[k] = 0
    else:
      let d = a.shape[k - pad]
      if d == shape[k]: strides[k] = a.strides[k - pad]
      elif d == 1: strides[k] = 0
      else:
        raise newException(ValueError,
          "cannot broadcast " & $a.shape & " to " & $(@shape))
  view(a, @shape, strides, a.offset)

proc broadcast2*[T, U](a: NDArray[T], b: NDArray[U]): (NDArray[T], NDArray[U]) =
  ## Both operands stretched to their common shape. Every binary operation
  ## starts here, so broadcasting is defined in exactly one place.
  let shape = broadcastShapes(a.shape, b.shape)
  (a.broadcastTo(shape), b.broadcastTo(shape))

# ----------------------------------------------------------- joining -------

proc concat*[T](arrays: openArray[NDArray[T]], axis = 0): NDArray[T] =
  ## Join along an existing axis. Every other axis must match.
  if arrays.len == 0:
    raise newException(ValueError, "concat needs at least one array")
  let nd = arrays[0].ndim
  let ax = normAxis(axis, nd)
  var shape = arrays[0].shape
  shape[ax] = 0
  for a in arrays:
    if a.ndim != nd:
      raise newException(ValueError, "concat: mixed dimensionality")
    for i in 0 ..< nd:
      if i != ax and a.shape[i] != arrays[0].shape[i]:
        raise newException(ValueError,
          "concat: shapes " & $arrays[0].shape & " and " & $a.shape &
          " differ on axis " & $i)
    shape[ax] += a.shape[ax]
  result = newNDArray[T](shape)
  var at = 0
  for a in arrays:
    # write each source into the slot it owns along `ax`, elementwise so that
    # a non-contiguous source needs no temporary
    var idx = newSeq[int](nd)
    for off in a.offsets:
      var dst = idx
      dst[ax] += at
      result.setElem(dst, a.buf[off])
      for k in countdown(nd - 1, 0):
        idx[k] += 1
        if idx[k] < a.shape[k]: break
        idx[k] = 0
    at += a.shape[ax]

proc stack*[T](arrays: openArray[NDArray[T]], axis = 0): NDArray[T] =
  ## Join along a **new** axis; every array must have the same shape.
  if arrays.len == 0:
    raise newException(ValueError, "stack needs at least one array")
  var expanded = newSeq[NDArray[T]](arrays.len)
  for i, a in arrays:
    if a.shape != arrays[0].shape:
      raise newException(ValueError,
        "stack: shapes " & $arrays[0].shape & " and " & $a.shape & " differ")
    expanded[i] = a.expandDims(axis)
  concat(expanded, axis)

proc vstack*[T](arrays: openArray[NDArray[T]]): NDArray[T] =
  ## Stack row-wise. 1-d inputs are treated as rows, as in numpy.
  var rows = newSeq[NDArray[T]](arrays.len)
  for i, a in arrays:
    rows[i] = if a.ndim == 1: a.reshape(1, a.size) else: a
  concat(rows, 0)

proc hstack*[T](arrays: openArray[NDArray[T]]): NDArray[T] =
  ## Stack column-wise; for 1-d inputs this is plain concatenation.
  if arrays.len > 0 and arrays[0].ndim == 1: concat(arrays, 0)
  else: concat(arrays, 1)

# ------------------------------------------------------------- casting -----

proc astype*[T, U](a: NDArray[T], _: typedesc[U]): NDArray[U] =
  ## Elementwise conversion to another element type, always a copy.
  ## `bool` counts as 0/1 for numeric targets, matching numpy.
  result = newNDArray[U](a.shape)
  var i = 0
  for x in a:
    when T is bool and U is SomeNumber:
      result.buf[i] = (if x: U(1) else: U(0))
    elif U is bool and T is SomeNumber:
      result.buf[i] = x != T(0)
    else:
      result.buf[i] = U(x)
    i += 1
