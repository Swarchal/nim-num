## Selecting parts of an array.
##
## `a[1, All]`, `a[0..2, 1..^1]`, `a[span(0, 8, 2)]` — one `Sel` per axis,
## reached through implicit converters so an ordinary `int` or Nim slice can
## be written where a `Sel` is expected. Missing trailing axes are taken
## whole, as in numpy, so `a[1]` is `a[1, All]` on a 2-d array.
##
## A slice is a **view**: it shares the buffer, so writing through it writes
## into the parent. `a[0..1].copy()` is how you get a detached one. An
## integer selection **drops** that axis (`a[1]` of a 2-d array is 1-d);
## a one-element span keeps it (`a[1..1]` stays 2-d). That distinction is
## numpy's, and it is what makes `a[i, j]` a scalar.
##
## `_` cannot be an identifier in Nim, so numpy's `:` is spelled `All`.

import ./core
import ./shape

type
  SelKind* = enum
    selAll
    selIndex
    selSpan

  Sel* = object
    ## One axis's worth of selection.
    case kind*: SelKind
    of selAll: discard
    of selIndex:
      idx*: int
      idxFromEnd*: bool
    of selSpan:
      first*, last*, step*: int
      firstFromEnd*, lastFromEnd*: bool

const All* = Sel(kind: selAll)
  ## The whole axis — numpy's bare `:`.

func span*(first, last: int, step = 1): Sel =
  ## An explicit strided run, both ends **inclusive** (as Nim's `..` and
  ## racoon's slices are). A negative step runs backwards.
  if step == 0:
    raise newException(ValueError, "span: step must be non-zero")
  Sel(kind: selSpan, first: first, last: last, step: step)

func toSel*(s: Sel): Sel = s
  ## Identity, so the `[]=` macro can wrap every index in `toSel` without
  ## caring whether it was already one.

converter toSel*(i: int): Sel = Sel(kind: selIndex, idx: i)
converter toSel*(i: BackwardsIndex): Sel =
  Sel(kind: selIndex, idx: int(i), idxFromEnd: true)
converter toSel*(s: HSlice[int, int]): Sel =
  Sel(kind: selSpan, first: s.a, last: s.b, step: 1)
converter toSel*(s: HSlice[int, BackwardsIndex]): Sel =
  Sel(kind: selSpan, first: s.a, last: int(s.b), step: 1, lastFromEnd: true)
converter toSel*(s: HSlice[BackwardsIndex, int]): Sel =
  Sel(kind: selSpan, first: int(s.a), last: s.b, step: 1, firstFromEnd: true)
converter toSel*(s: HSlice[BackwardsIndex, BackwardsIndex]): Sel =
  Sel(kind: selSpan, first: int(s.a), last: int(s.b), step: 1,
      firstFromEnd: true, lastFromEnd: true)

func resolve(pos: int, fromEnd: bool, n: int): int =
  ## The one place `^k` and a plain index become a position. Nim resolves
  ## `^k` inside a template private to `system`, so a container that is not
  ## `seq`/`array` has to restate it — racoon does the same.
  if fromEnd: n - pos else: pos

func applySel(s: Sel, n, stride: int): tuple[len, stride, offset: int, drop: bool] =
  ## What one selection does to one axis: its new length and stride, the
  ## offset it contributes, and whether the axis disappears.
  case s.kind
  of selAll:
    (n, stride, 0, false)
  of selIndex:
    let i = resolve(s.idx, s.idxFromEnd, n)
    if i < 0 or i >= n:
      raise newException(ValueError,
        "index " & $s.idx & " out of range for an axis of length " & $n)
    (0, 0, i * stride, true)
  of selSpan:
    let a = resolve(s.first, s.firstFromEnd, n)
    let b = resolve(s.last, s.lastFromEnd, n)
    # `a ..< a` is the empty slice at any `a`, following seq[T]; anything
    # else running past either end is an error rather than a silent clamp.
    if s.step > 0 and b == a - 1:
      (0, s.step * stride, 0, false)
    else:
      if a < 0 or a >= n or b < 0 or b >= n:
        raise newException(ValueError,
          "slice " & $s.first & ".." & $s.last & " out of range for an axis of length " & $n)
      let count = if s.step > 0:
                    (if b < a: 0 else: (b - a) div s.step + 1)
                  else:
                    (if b > a: 0 else: (a - b) div (-s.step) + 1)
      (count, s.step * stride, a * stride, false)

proc select*[T](a: NDArray[T], sels: varargs[Sel]): NDArray[T] =
  ## A view selected axis by axis. Trailing axes not mentioned are taken
  ## whole. Users reach this through `[]`.
  if sels.len > a.ndim:
    raise newException(ValueError,
      $sels.len & " selections for a " & $a.ndim & "-d array")
  var shape, strides: seq[int]
  var offset = a.offset
  for ax in 0 ..< a.ndim:
    let s = if ax < sels.len: sels[ax] else: All
    let (l, st, off, drop) = applySel(s, a.shape[ax], a.strides[ax])
    offset += off
    if not drop:
      shape.add(l)
      strides.add(st)
  view(a, shape, strides, offset)

proc setSelect*[T](a: var NDArray[T], sels: openArray[Sel], val: T) =
  ## Write one value into every element the selection covers.
  var target = a.select(sels)
  target.fill(val)

proc setSelect*[T](a: var NDArray[T], sels: openArray[Sel], src: NDArray[T]) =
  ## Write `src` into the selection, broadcasting it to fit.
  var target = a.select(sels)
  # `a[1..3] = a[0..2]` reads cells the loop has already written, so a source
  # sharing the destination's buffer is materialised first. numpy makes the
  # same guarantee: the right-hand side is the array as it was.
  let source = if src.buf == a.buf: src.copy() else: src
  let rhs = source.broadcastTo(target.shape)
  for ot, os in offsets2(target, rhs): a.buf[ot] = rhs.buf[os]

# ------------------------------------------------------ fancy indexing -----

proc take*[T](a: NDArray[T], indices: openArray[int], axis = 0): NDArray[T] =
  ## Gather positions along one axis, in the given order and with repeats
  ## allowed — the row-gather primitive, and always a copy. Racoon's
  ## `selectRow(seq[int])` is the same idea for frames.
  let ax = normAxis(axis, a.ndim)
  var shape = a.shape
  shape[ax] = indices.len
  result = newNDArray[T](shape)
  var sels = newSeq[Sel](a.ndim)
  for i in 0 ..< a.ndim: sels[i] = All
  for k, i in indices:
    let j = if i < 0: i + a.shape[ax] else: i
    if j < 0 or j >= a.shape[ax]:
      raise newException(ValueError,
        "index " & $i & " out of range for axis " & $ax & " of length " & $a.shape[ax])
    sels[ax] = span(j, j)
    var dstSel = sels
    dstSel[ax] = span(k, k)
    result.setSelect(dstSel, a.select(sels))

proc select*[T](a: NDArray[T], mask: NDArray[bool]): NDArray[T] =
  ## Boolean selection: the elements where `mask` is true, flattened to 1-d,
  ## as numpy's `a[a > 0]` gives. A copy — the survivors are not a strided
  ## window in general.
  if mask.shape != a.shape:
    raise newException(ValueError,
      "mask shape " & $mask.shape & " does not match array shape " & $a.shape)
  var vals: seq[T]
  for oa, om in offsets2(a, mask):
    if mask.buf[om]: vals.add(a.buf[oa])
  initNDArray[T](@[vals.len], vals)

proc setSelect*[T](a: var NDArray[T], mask: NDArray[bool], val: T) =
  ## Write `val` wherever the mask is true, leaving the rest alone.
  if mask.shape != a.shape:
    raise newException(ValueError,
      "mask shape " & $mask.shape & " does not match array shape " & $a.shape)
  for oa, om in offsets2(a, mask):
    if mask.buf[om]: a.buf[oa] = val

proc nonZero*(mask: NDArray[bool]): seq[int] =
  ## Flat positions where the mask is true — the indices `take` wants.
  var i = 0
  for x in mask:
    if x: result.add(i)
    i.inc

proc row*[T](a: NDArray[T], i: int): NDArray[T] =
  ## Row `i` of a 2-d array as a 1-d view.
  if a.ndim != 2: raise newException(ValueError, "row() takes a 2-d array")
  a.select(i, All)

proc col*[T](a: NDArray[T], j: int): NDArray[T] =
  ## Column `j` of a 2-d array as a 1-d view (stride = the row length).
  if a.ndim != 2: raise newException(ValueError, "col() takes a 2-d array")
  a.select(All, j)
