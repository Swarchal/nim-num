import std/[unittest, math]
import num

suite "arithmetic":
  test "elementwise and scalar forms agree":
    let a = toNDArray(@[1.0, 2.0, 3.0])
    check (a + a).toSeq() == @[2.0, 4.0, 6.0]
    check (a + 1.0).toSeq() == @[2.0, 3.0, 4.0]
    check (1.0 + a).toSeq() == @[2.0, 3.0, 4.0]
    check (a * 2.0).toSeq() == @[2.0, 4.0, 6.0]
    check (10.0 - a).toSeq() == @[9.0, 8.0, 7.0]
    check (-a).toSeq() == @[-1.0, -2.0, -3.0]

  test "division of ints gives floats, div truncates":
    let a = toNDArray(@[1, 2, 3])
    check (a / 2).toSeq() == @[0.5, 1.0, 1.5]
    check (a div 2).toSeq() == @[0, 1, 1]
    check (a mod 2).toSeq() == @[1, 0, 1]

  test "binary ops broadcast":
    let a = arange(6).reshape(2, 3)
    let b = arange(3)
    check (a + b).toSeq() == @[0, 2, 4, 3, 5, 7]
    let col = arange(2).reshape(2, 1)
    check (a + col).toSeq() == @[0, 1, 2, 4, 5, 6]
    expect ValueError: discard a + arange(4)

  test "in-place ops write through a view":
    var a = zeros[float](2, 3)
    var row = a[0, All]
    row += 5.0
    check a[0, 1] == 5.0
    check a[1, 1] == 0.0
    a += ones[float](2, 3)
    check a[1, 1] == 1.0

suite "comparisons":
  test "give bool arrays, elementwise":
    let a = arange(4)
    check (a > 1).toSeq() == @[false, false, true, true]
    check (a.eq 2).toSeq() == @[false, false, true, false]
    check (a.neq 2).toSeq() == @[true, true, false, true]
    check ((a > 0) and (a < 3)).toSeq() == @[false, true, true, false]
    check (not (a > 0)).toSeq() == @[true, false, false, false]

  test "== is whole-array, not elementwise":
    check arange(3) == arange(3)
    check not (arange(3) == arange(4))

  test "allclose tolerates float noise":
    let a = toNDArray(@[0.1 + 0.2, 1.0])
    check allclose(a, toNDArray(@[0.3, 1.0]))
    check not allclose(a, toNDArray(@[0.4, 1.0]))
    check not allclose(toNDArray(@[NaN]), toNDArray(@[NaN]))

suite "math":
  test "ufuncs":
    check allclose(sqrt(toNDArray(@[4.0, 9.0])), toNDArray(@[2.0, 3.0]))
    check allclose(exp(zeros[float](2)), ones[float](2))
    check abs(toNDArray(@[-1, 2])).toSeq() == @[1, 2]
    check square(toNDArray(@[2, 3])).toSeq() == @[4, 9]
    check sign(toNDArray(@[-2.0, 0.0, 5.0])).toSeq() == @[-1.0, 0.0, 1.0]
    check allclose(pow(toNDArray(@[2.0, 3.0]), 2.0), toNDArray(@[4.0, 9.0]))

  test "maximum is elementwise, max is the fold":
    let a = toNDArray(@[1, 5, 3])
    let b = toNDArray(@[4, 2, 3])
    check maximum(a, b).toSeq() == @[4, 5, 3]
    check minimum(a, 3).toSeq() == @[1, 3, 3]
    check max(a) == 5

  test "clip":
    check clip(arange(5), 1, 3).toSeq() == @[1, 1, 2, 3, 3]

  test "isNaN and isFinite":
    let a = toNDArray(@[1.0, NaN, Inf])
    check a.isNaN.toSeq() == @[false, true, false]
    check a.isFinite.toSeq() == @[true, false, false]

suite "where and mapping":
  test "where picks elementwise":
    let a = arange(4)
    check where(a > 1, a, zeros[int](4)).toSeq() == @[0, 0, 2, 3]
    check where(a > 1, a, -1).toSeq() == @[-1, -1, 2, 3]

  test "mapIt decides its own element type":
    let a = arange(4)
    check a.mapIt(it * 2).toSeq() == @[0, 2, 4, 6]
    check a.mapIt(it.float / 2.0).toSeq() == @[0.0, 0.5, 1.0, 1.5]
    check a.mapIt(it mod 2 == 0).toSeq() == @[true, false, true, false]

  test "zipIt sees x and y":
    check zipIt(arange(3), arange(3), x * y).toSeq() == @[0, 1, 4]
