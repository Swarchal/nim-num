import std/unittest
import std/sequtils
import num

suite "selection":
  let a = arange(12).reshape(3, 4)

  test "an int drops the axis, a span keeps it":
    check a[1, All].shape == @[4]
    check a[1, All].toSeq() == @[4, 5, 6, 7]
    check a[1..1, All].shape == @[1, 4]
    check a[All, 2].toSeq() == @[2, 6, 10]

  test "trailing axes are taken whole, but an all-int index must name them":
    check a[0..1].shape == @[2, 4]
    check a[1..1].shape == @[1, 4]
    expect ValueError: discard a[0, 0, 0]
    # `a[1]` is an *element* access — it cannot be a row, because whether it
    # names every axis is a runtime fact. Write `a[1, All]` or `a.row(1)`.
    expect ValueError: discard a.elemAt(1)

  test "slices, backwards indices and steps":
    check a[0..1, 1..2].toSeq() == @[1, 2, 5, 6]
    check a[^1, All].toSeq() == @[8, 9, 10, 11]
    check a[0, 1..^2].toSeq() == @[1, 2]
    check a[All, span(0, 3, 2)].toSeq() == @[0, 2, 4, 6, 8, 10]
    check a[0, span(3, 0, -1)].toSeq() == @[3, 2, 1, 0]
    check a[0, 1..0].size == 0                # `a ..< a` is empty
    expect ValueError: discard a[0, 1..9]
    expect ValueError: discard span(0, 3, 0)

  test "row and col are 2-d shorthands":
    check a.row(1) == a[1, All]
    check a.col(1) == a[All, 1]

  test "slices are views":
    var b = arange(12).reshape(3, 4)
    var win = b[1..2, 1..2]
    win.fill(0)
    check b[1, 1] == 0
    check b[0, 1] == 1

suite "Rest":
  let a = arange(24).reshape(2, 3, 4)

  test "it covers the axes the others leave, so a trailing index is the last axis":
    check a[Rest, 1].shape == @[2, 3]
    check a[Rest, 1] == a[All, All, 1]
    check a[1, Rest].shape == @[3, 4]
    check a[1, Rest] == a[1, All, All]
    check a[0, Rest, 2] == a[0, All, 2]
    check a[Rest, 1..2].shape == @[2, 3, 2]

  test "it is not All":
    check a[All, 1].shape == @[2, 4]
    check a[Rest, 1].shape == @[2, 3]

  test "it may stand for no axes at all":
    check a[Rest] == a
    check a[0, 1, Rest, 2].item == a[0, 1, 2]

  test "it is a Sel, so an all-other-int index still gives a 0-d view":
    let v = arange(5)
    check v[Rest, 3].ndim == 0
    check v[Rest, 3].item == 3
    check v[Rest, ^1].item == 4

  test "at most one, and it does not excuse too many selections":
    expect ValueError: discard a[Rest, 0, Rest]
    expect ValueError: discard a[0, 0, 0, 0, Rest]

  test "assignment through it":
    var b = zeros[int](2, 3, 4)
    b[Rest, 0] = 7
    check b[All, All, 0].toSeq().allIt(it == 7)
    check sum(b) == 7 * 6
    b[1, Rest] = ones[int](4)
    check b[1, 2, 3] == 1
    check b[0, 2, 3] == 0

suite "assignment":
  test "a scalar fills the selection":
    var a = zeros[int](3, 3)
    a[1, All] = 5
    check a.row(1).toSeq() == @[5, 5, 5]
    check a.row(0).toSeq() == @[0, 0, 0]

  test "an array is broadcast into it":
    var a = zeros[int](3, 3)
    a[All, 0] = toNDArray(@[1, 2, 3])
    check a.col(0).toSeq() == @[1, 2, 3]
    a[0..1, All] = ones[int](2, 3)
    check a[0, 2] == 1

  test "single elements":
    var a = zeros[float](2, 2)
    a[1, 1] = 3.0
    check a[1, 1] == 3.0

suite "boolean and fancy indexing":
  let a = arange(6).reshape(2, 3)

  test "a mask selects flat":
    check a[a > 2].toSeq() == @[3, 4, 5]
    check a[(a mod 2).eq(0)].toSeq() == @[0, 2, 4]
    expect ValueError: discard a[toNDArray(@[true, false])]

  test "masked assignment leaves the rest alone":
    var b = arange(6)
    b[b > 3] = 0
    check b.toSeq() == @[0, 1, 2, 3, 0, 0]

  test "^k selects rather than indexing, and item unwraps it":
    let v = linspace(0.0, 1.0, 5)
    check v[^1] is NDArray[float]
    check v[0] is float
    check v[^1].item == 1.0
    check v.toSeq()[^1] == 1.0

  test "take gathers along an axis, repeats allowed":
    check a.take(@[1, 0], axis = 0).toSeq() == @[3, 4, 5, 0, 1, 2]
    check a.take(@[0, 0], axis = 1).toSeq() == @[0, 0, 3, 3]
    expect ValueError: discard a.take(@[9], axis = 0)

  test "nonZero gives the positions a mask marks":
    check (arange(5) > 2).nonZero == @[3, 4]

suite "overlapping assignment":
  test "a slice assigned from its own array reads the old values":
    var a = arange(5)
    a[1..3] = a[0..2]
    check a.toSeq() == @[0, 0, 1, 2, 4]
    var b = arange(5)
    b[0..2] = b[1..3]
    check b.toSeq() == @[1, 2, 3, 3, 4]

  test "compound assignment through overlapping views":
    var a = arange(5)
    var v = a[1..3]
    v += a[0..2]
    check a.toSeq() == @[0, 1, 3, 5, 4]
    var b = arange(5)
    var w = b[0..2]
    w -= b[1..3]
    check b.toSeq() == @[-1, -1, -1, 3, 4]
