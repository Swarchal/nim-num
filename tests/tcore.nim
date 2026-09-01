import std/[unittest, strutils]
import num

suite "construction and geometry":
  test "shape, size, ndim":
    let a = zeros[float](2, 3)
    check a.shape == @[2, 3]
    check a.size == 6
    check a.ndim == 2
    check a.len == 2                 # first axis, not element count
    check a.isContiguous

  test "0-d arrays are scalars":
    let s = scalarArray(4.0)
    check s.ndim == 0
    check s.size == 1
    check s.item == 4.0

  test "toNDArray from nested seqs":
    let a = toNDArray(@[@[1, 2, 3], @[4, 5, 6]])
    check a.shape == @[2, 3]
    check a[1, 2] == 6
    expect ValueError:
      discard toNDArray(@[@[1, 2], @[3]])

  test "arange and linspace":
    check arange(5).toSeq() == @[0, 1, 2, 3, 4]
    check arange(1, 7, 2).toSeq() == @[1, 3, 5]
    check arange(5, 0, -2).toSeq() == @[5, 3, 1]
    check arange(0.0, 1.0, 0.25).size == 4
    # every point is computed from `start`: accumulating `step` drifted, and
    # this used to end at 0.8999999999999999
    check arange(0.0, 1.0, 0.1).toSeq()[^1] == 0.9
    check arange(1.0, 0.0, -0.25).toSeq() == @[1.0, 0.75, 0.5, 0.25]
    let l = linspace(0.0, 1.0, 5)
    check l.toSeq() == @[0.0, 0.25, 0.5, 0.75, 1.0]
    check linspace(0.0, 1.0, 5)[4] == 1.0        # the endpoint is exact
    check linspace(0.0, 1.0, 4, endpoint = false).toSeq() == @[0.0, 0.25, 0.5, 0.75]
    check linspace(2.0, 3.0, 1).toSeq() == @[2.0]

  test "logspace spaces the exponents, endpoint and all":
    check logspace(0.0, 3.0, 4).toSeq() == @[1.0, 10.0, 100.0, 1000.0]
    # `endpoint` used to be missing here, so this was unaskable
    check logspace(0.0, 2.0, 2, endpoint = false).toSeq() == @[1.0, 10.0]
    check logspace(0.0, 3.0, 4, base = 2.0).toSeq() == @[1.0, 2.0, 4.0, 8.0]

  test "eye and full":
    let i3 = eye[float](3)
    check i3[0, 0] == 1.0
    check i3[0, 1] == 0.0
    check eye[int](2, 3, 1).toSeq() == @[0, 1, 0, 0, 0, 1]
    check full(@[2, 2], 7).toSeq() == @[7, 7, 7, 7]

  test "element access needs every axis":
    let a = arange(6).reshape(2, 3)
    check a[1, 1] == 4
    check a[1, -1] == 5                          # negative indices from the end
    expect ValueError:
      discard a.elemAt(1)                        # not the first row: an error
    expect ValueError:
      discard a[0, 3]

  test "copies detach, views do not":
    var a = zeros[int](2, 2)
    let v = a[0, All]
    let c = a[0, All].copy()
    a[0, 0] = 9
    check v[0] == 9
    check c[0] == 0

suite "equality and printing":
  test "structural equality ignores layout":
    let a = arange(6).reshape(2, 3)
    check a == a.copy()
    check a.t.t == a
    check a != arange(6).reshape(3, 2)

  test "$ renders nested rows":
    check $arange(4).reshape(2, 2) == "array([[0, 1],\n       [2, 3]])"
    check $scalarArray(1) == "array(1)"

  test "floats print to six significant digits, trimmed":
    check $toNDArray(@[1.0, 0.5]) == "array([1.0, 0.5])"
    check $toNDArray(@[2.718281828459045]) == "array([2.71828])"
    check $toNDArray(@[1e-12]) == "array([1.0e-12])"
    check $toNDArray(@[NaN, Inf]) == "array([nan, inf])"

  test "an empty array prints its shape the way describe does":
    check $zeros[int](0, 3) == "array([], shape=(0, 3))"
    check $zeros[int](0) == "array([], shape=(0))"

  test "describe":
    check arange(6).reshape(2, 3).describe ==
      "NDArray[int] shape=(2, 3) strides=(3, 1) contiguous"

suite "negative strides":
  # `span(first, last, step)` with a negative step is the one thing that
  # builds a view with a negative stride, and `offsets` claims to walk any
  # strides at all. Everything generic in the library is written over that
  # claim, so it is exercised here rather than trusted.
  let a = arange(12).reshape(3, 4)         # [[0..3],[4..7],[8..11]]
  let rowRev = a[span(2, 0, -1), All]      # strides (-4, 1)
  let colRev = a[All, span(3, 0, -1)]      # strides ( 4,-1)
  let bothRev = a[span(2, 0, -1), span(3, 0, -1)]

  test "the traversal visits a reversed view in logical order":
    check rowRev.toSeq() == @[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    check colRev.toSeq() == @[3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
    check bothRev.toSeq() == @[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    check a[All, span(3, 0, -2)].toSeq() == @[3, 1, 7, 5, 11, 9]

  test "a reversed view keeps its shape and loses contiguity":
    check rowRev.shape == @[3, 4]
    check rowRev.strides == @[-4, 1]
    check not rowRev.isContiguous
    check not colRev.isContiguous

  test "element access resolves through a negative stride":
    check rowRev[0, 0] == 8
    check rowRev[2, 3] == 3
    check colRev[0, 0] == 3
    check bothRev[0, 0] == 11
    expect ValueError: discard rowRev[3, 0]

  test "copying a reversed view detaches it and makes it contiguous":
    let c = rowRev.copy()
    check c.isContiguous
    check c.toSeq() == rowRev.toSeq()
    check rowRev.asContiguous().toSeq() == rowRev.toSeq()
    check c == rowRev                       # equality ignores layout

  test "$ prints a reversed view in logical order":
    check $arange(4)[span(3, 0, -1)] == "array([3, 2, 1, 0])"
    check $rowRev == "array([[ 8,  9, 10, 11],\n" &
                     "       [ 4,  5,  6,  7],\n" &
                     "       [ 0,  1,  2,  3]])"

  test "reshape and transpose of a reversed view read its logical order":
    check rowRev.reshape(12).toSeq() == rowRev.toSeq()
    check rowRev.ravel().toSeq() == rowRev.toSeq()
    check rowRev.t.shape == @[4, 3]
    check rowRev.t.toSeq() == @[8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3]

  test "the two-array traversal lines up views with unlike strides":
    check (rowRev + colRev).toSeq() == @[11, 11, 11, 11, 11, 11,
                                         11, 11, 11, 11, 11, 11]
    check (rowRev - a).toSeq() == @[8, 8, 8, 8, 0, 0, 0, 0, -8, -8, -8, -8]
    check (colRev * 2).toSeq() == @[6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16]

  test "broadcasting a stride-0 axis against a negative one":
    let colVec = arr(@[@[100], @[200], @[300]])
    check (rowRev + colVec).toSeq() == @[108, 109, 110, 111,
                                         204, 205, 206, 207,
                                         300, 301, 302, 303]

  test "order-sensitive reductions respect the reversed order":
    check cumsum(arange(5)[span(4, 0, -1)]).toSeq() == @[4, 7, 9, 10, 10]
    check diff(arange(5)[span(4, 0, -1)]).toSeq() == @[-1, -1, -1, -1]
    check sum(colRev, axis = 0).toSeq() == @[21, 18, 15, 12]
    check sum(rowRev, axis = 1).toSeq() == @[38, 22, 6]
    check argmax(colRev) == 8               # a flat index into the view
    check argmin(rowRev) == 8

  test "gathering and masking read a reversed view correctly":
    check take(rowRev, @[0, 2], axis = 0).toSeq() == @[8, 9, 10, 11, 0, 1, 2, 3]
    check colRev[colRev > 8].toSeq() == @[11, 10, 9]

  test "matmul copies a reversed operand before walking it":
    let m = arange(4).reshape(2, 2)
    let mrev = m[All, span(1, 0, -1)]       # [[1, 0], [3, 2]]
    check matmul(mrev, identity[int](2)).toSeq() == @[1, 0, 3, 2]
    check matmul(identity[int](2), mrev).toSeq() == @[1, 0, 3, 2]

  test "writing through a reversed view reaches the base array":
    var b = arange(12).reshape(3, 4)
    var rev = b[All, span(3, 0, -1)]
    rev[0, 0] = 99
    check b[0, 3] == 99
    rev.fill(0)
    check b.toSeq() == @[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  test "assigning into a negative-step selection":
    var c = arange(5)
    c[span(4, 2, -1)] = 0
    check c.toSeq() == @[0, 1, 0, 0, 0]
    var d = arange(5)
    d[span(4, 0, -1)] = arange(5)
    check d.toSeq() == @[4, 3, 2, 1, 0]

  test "reversing in place reads the array as it was":
    var v = arange(5)
    v[All] = v[span(4, 0, -1)]
    check v.toSeq() == @[4, 3, 2, 1, 0]

suite "truncated printing":
  test "a small array still prints in full":
    check $arange(6) == "array([0, 1, 2, 3, 4, 5])"
    check ($arange(1000)).count("...") == 0

  test "past the threshold only the corners are shown":
    let s = $arange(2000)
    check s == "array([   0,    1,    2,  ..., 1997, 1998, 1999])"

  test "every axis of a large array is summarised":
    let s = $arange(2500).reshape(50, 50)
    check s.splitLines().len == 7             # 3 rows, the gap, 3 rows
    check s.splitLines()[0] == "array([[   0,    1,    2,  ...,   47,   48,   49],"
    check s.splitLines()[3] == "       ...,"
    check s.splitLines()[6] == "       [2450, 2451, 2452,  ..., 2497, 2498, 2499]])"

  test "the width is measured over what is printed, not the whole array":
    # the one wide element sits in the hidden middle: measuring every element
    # would pad all six visible zeros out to its width for nothing
    var wide = zeros[float](2000)
    wide[1000] = 123456789.0
    check $wide == "array([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])"

  test "truncation reads a view in logical order too":
    check $flip(arange(2000)) == "array([1999, 1998, 1997,  ...,    2,    1,    0])"
