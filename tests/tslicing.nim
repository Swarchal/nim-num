import std/unittest
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
