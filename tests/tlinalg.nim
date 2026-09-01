import std/[unittest, math]
import num

suite "products":
  test "dot is the vector inner product":
    check dot(toNDArray(@[1.0, 2.0, 3.0]), toNDArray(@[4.0, 5.0, 6.0])) == 32.0
    expect ValueError: discard dot(arange(3), arange(4))

  test "matmul of two matrices":
    let a = toNDArray(@[@[1.0, 2.0], @[3.0, 4.0]])
    let b = toNDArray(@[@[5.0, 6.0], @[7.0, 8.0]])
    check matmul(a, b).toSeq() == @[19.0, 22.0, 43.0, 50.0]
    check matmul(a, eye[float](2)) == a
    expect ValueError: discard matmul(a, zeros[float](3, 3))

  test "a 1-d operand becomes a row or a column and is dropped again":
    let a = toNDArray(@[@[1.0, 2.0], @[3.0, 4.0]])
    let v = toNDArray(@[1.0, 1.0])
    check matmul(a, v).shape == @[2]
    check matmul(a, v).toSeq() == @[3.0, 7.0]
    check matmul(v, a).toSeq() == @[4.0, 6.0]
    check matmul(v, v).ndim == 0

  test "matmul works on a transposed view":
    let a = arange(6).reshape(2, 3).astype(float)
    check matmul(a, a.t).shape == @[2, 2]
    check matmul(a, a.t).toSeq() == @[5.0, 14.0, 14.0, 50.0]

  test "outer, trace, diag":
    check outer(toNDArray(@[1.0, 2.0]), toNDArray(@[3.0, 4.0])).toSeq() ==
      @[3.0, 4.0, 6.0, 8.0]
    check trace(eye[float](3)) == 3.0
    check diag(arange(4).reshape(2, 2)).toSeq() == @[0, 3]
    check diag(toNDArray(@[1, 2])).toSeq() == @[1, 0, 0, 2]

  test "norm":
    check norm(toNDArray(@[3.0, 4.0])) == 5.0
    check norm(toNDArray(@[3.0, -4.0]), 1.0) == 7.0
    check norm(toNDArray(@[3.0, -4.0]), Inf) == 4.0
    check norm(toNDArray(@[3.0, -4.0]), NegInf) == 3.0
    # the three numpy spells out are limits of the formula, not values of
    # it: `pow(s, 1/p)` returned inf here whatever the input
    check norm(toNDArray(@[3.0, 0.0, -4.0]), 0.0) == 2.0
    check norm(zeros[float](3), 0.0) == 0.0
    # a finite negative p is the ordinary formula, as it is in numpy, and
    # -Inf above is its limit — accepting one and rejecting the other was
    # the odd rule
    check abs(norm(toNDArray(@[3.0, -4.0]), -1.0) - 12.0 / 7.0) < 1e-12
    check abs(norm(toNDArray(@[3.0, -4.0]), -2.0) - 2.4) < 1e-12
    check norm(toNDArray(@[3.0, 0.0, -4.0]), -1.0) == 0.0   # 1/0 dominates

suite "solving":
  let a = toNDArray(@[@[2.0, 1.0], @[1.0, 3.0]])

  test "solve recovers the right-hand side":
    let x = solve(a, toNDArray(@[5.0, 10.0]))
    check allclose(matmul(a, x), toNDArray(@[5.0, 10.0]))

  test "solve handles several right-hand sides at once":
    let b = toNDArray(@[@[5.0, 1.0], @[10.0, 0.0]])
    let x = solve(a, b)
    check x.shape == @[2, 2]
    check allclose(matmul(a, x), b)

  test "inv and det":
    check allclose(matmul(a, inv(a)), eye[float](2))
    check abs(det(a) - 5.0) < 1e-12
    check det(zeros[float](2, 2)) == 0.0

  test "pivoting keeps a zero-pivot matrix solvable":
    let m = toNDArray(@[@[0.0, 1.0], @[1.0, 0.0]])
    check allclose(solve(m, toNDArray(@[1.0, 2.0])), toNDArray(@[2.0, 1.0]))

  test "a singular matrix raises":
    expect ValueError: discard solve(toNDArray(@[@[1.0, 2.0], @[2.0, 4.0]]),
                                     toNDArray(@[1.0, 2.0]))
    expect ValueError: discard solve(zeros[float](2, 3), toNDArray(@[1.0, 2.0]))

  test "lstsq fits a line":
    # y = 2x + 1 exactly, so the least-squares fit is the line itself
    let x = toNDArray(@[0.0, 1.0, 2.0, 3.0])
    let design = hstack(@[x.reshape(4, 1), ones[float](4, 1)])
    let y = x * 2.0 + 1.0
    check allclose(lstsq(design, y), toNDArray(@[2.0, 1.0]))

  test "lstsq minimises the residual on an inexact fit":
    # the four points are not collinear, so this is a real projection: the
    # residual must be orthogonal to both columns of the design matrix
    let x = toNDArray(@[0.0, 1.0, 2.0, 3.0])
    let design = hstack(@[x.reshape(4, 1), ones[float](4, 1)])
    let y = toNDArray(@[1.0, 3.0, 2.0, 5.0])
    let coef = lstsq(design, y)
    check allclose(coef, toNDArray(@[1.1, 1.1]))
    let resid = y - matmul(design, coef)
    check allclose(matmul(design.t, resid), zeros[float](2), atol = 1e-12)

  test "lstsq survives a design matrix the normal equations cannot":
    # Lauchli: A'A is [[1+eps^2, 1], [1, 1+eps^2]], and 1 + 1e-16 is exactly
    # 1.0 in float64 — so forming A'A makes an invertible problem singular.
    # QR never forms it. The exact answer is 1/(2 + eps^2).
    const eps = 1e-8
    let a = toNDArray(@[@[1.0, 1.0], @[eps, 0.0], @[0.0, eps]])
    let b = toNDArray(@[1.0, 0.0, 0.0])
    check 1.0 + eps * eps == 1.0                 # the precision that is lost
    expect ValueError:                           # the old normal-equations route
      discard solve(matmul(a.t, a), matmul(a.t, b))
    check allclose(lstsq(a, b), toNDArray(@[0.5, 0.5]))

  test "lstsq takes several right-hand sides at once":
    let x = toNDArray(@[0.0, 1.0, 2.0, 3.0])
    let design = hstack(@[x.reshape(4, 1), ones[float](4, 1)])
    let ys = hstack(@[(x * 2.0 + 1.0).reshape(4, 1),
                      (x * -1.0 + 4.0).reshape(4, 1)])
    let coef = lstsq(design, ys)
    check coef.shape == @[2, 2]
    check allclose(coef, toNDArray(@[@[2.0, -1.0], @[1.0, 4.0]]))
    check allclose(lstsq(design, ys[All, 0..0]), toNDArray(@[@[2.0], @[1.0]]))

  test "lstsq rejects what it cannot solve":
    let wide = ones[float](2, 3)
    expect ValueError: discard lstsq(wide, toNDArray(@[1.0, 2.0]))
    # a repeated column is rank-deficient: R has a zero on its diagonal
    let dup = toNDArray(@[@[1.0, 1.0], @[2.0, 2.0], @[3.0, 3.0]])
    expect ValueError: discard lstsq(dup, toNDArray(@[1.0, 2.0, 3.0]))
    let design = ones[float](4, 2)
    expect ValueError: discard lstsq(design, toNDArray(@[1.0, 2.0]))
