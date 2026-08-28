import std/unittest
import num

suite "reshape":
  test "reshape is a view when contiguous":
    var a = arange(6)
    var b = a.reshape(2, 3)
    b[0, 0] = 99
    check a[0] == 99

  test "-1 is inferred":
    check arange(12).reshape(3, -1).shape == @[3, 4]
    check arange(12).reshape(-1).shape == @[12]
    expect ValueError: discard arange(12).reshape(-1, -1)
    expect ValueError: discard arange(12).reshape(5, -1)
    expect ValueError: discard arange(6).reshape(4, 2)

  test "reshape of a strided array copies":
    var a = arange(6).reshape(2, 3).t
    var b = a.reshape(6)
    check b.toSeq() == @[0, 3, 1, 4, 2, 5]
    b[0] = 99
    check a[0, 0] == 0                     # the copy is detached

suite "axes":
  test "transpose reverses by default":
    let a = arange(6).reshape(2, 3)
    check a.t.shape == @[3, 2]
    check a.t.toSeq() == @[0, 3, 1, 4, 2, 5]
    check a.transpose(1, 0) == a.t
    expect ValueError: discard a.transpose(0, 0)

  test "swapAxes and moveAxis":
    let a = zeros[int](2, 3, 4)
    check a.swapAxes(0, 2).shape == @[4, 3, 2]
    check a.moveAxis(0, 2).shape == @[3, 4, 2]

  test "expandDims and squeeze":
    let a = arange(6).reshape(2, 3)
    check a.expandDims(0).shape == @[1, 2, 3]
    check a.expandDims(-1).shape == @[2, 3, 1]
    check a.expandDims(1).squeeze(1) == a
    check zeros[int](1, 3, 1).squeeze().shape == @[3]
    expect ValueError: discard a.squeeze(0)

suite "broadcasting":
  test "the shape rule":
    check broadcastShapes(@[3, 1], @[1, 4]) == @[3, 4]
    check broadcastShapes(@[2, 3], @[3]) == @[2, 3]
    check broadcastShapes(@[], @[2, 2]) == @[2, 2]
    expect ValueError: discard broadcastShapes(@[2, 3], @[4])

  test "broadcastTo repeats with a zero stride":
    let a = arange(3).broadcastTo(@[2, 3])
    check a.shape == @[2, 3]
    check a.strides == @[0, 1]
    check a.toSeq() == @[0, 1, 2, 0, 1, 2]

suite "joining":
  test "concat along an axis":
    let a = arange(6).reshape(2, 3)
    check concat(@[a, a], 0).shape == @[4, 3]
    check concat(@[a, a], 1).toSeq() == @[0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]
    expect ValueError: discard concat(@[a, arange(4).reshape(2, 2)], 0)

  test "stack adds an axis":
    let a = arange(3)
    check stack(@[a, a], 0).shape == @[2, 3]
    check stack(@[a, a], 1).toSeq() == @[0, 0, 1, 1, 2, 2]

  test "vstack and hstack of 1-d":
    let a = arange(3)
    check vstack(@[a, a]).shape == @[2, 3]
    check hstack(@[a, a]).shape == @[6]

suite "casting":
  test "astype":
    check arange(3).astype(float).toSeq() == @[0.0, 1.0, 2.0]
    check toNDArray(@[true, false]).astype(int).toSeq() == @[1, 0]
    check toNDArray(@[0.0, 2.0]).astype(bool).toSeq() == @[false, true]

suite "flip":
  let a = arange(12).reshape(3, 4)

  test "flip reverses one axis and is a view":
    check flip(a, 0).toSeq() == @[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    check flip(a, 1).toSeq() == @[3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
    check flip(a, 0).shape == a.shape
    check flip(a, 0).strides == @[-4, 1]
    check not flip(a, 0).isContiguous

  test "flip with no axis turns the array around, unlike transpose":
    check flip(a).toSeq() == @[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    check flip(a).shape == @[3, 4]
    check flip(a) != a.t
    check flip(flip(a)) == a
    check flip(arange(4)).toSeq() == @[3, 2, 1, 0]

  test "a negative axis counts from the end":
    check flip(a, -1) == flip(a, 1)
    expect ValueError: discard flip(a, 2)

  test "flip aliases, like every other view here":
    var b = arange(6).reshape(2, 3)
    var f = flip(b, 1)
    f[0, 0] = 99
    check b[0, 2] == 99
    var g = flip(b, 0)          # `fill` takes a var, so the view is bound first
    g.fill(0)
    check b.toSeq() == @[0, 0, 0, 0, 0, 0]

  test "flip of an empty axis is empty, not an error":
    let e = zeros[int](0, 3)
    check flip(e, 0).size == 0
    check flip(e).shape == @[0, 3]

  test "flipping the same axis twice is the original":
    check flip(flip(a, 1), 1) == a
    check flip(a, 0).copy() == flip(a, 0)
