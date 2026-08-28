## Linear algebra, in plain Nim — no BLAS, no dependencies.
##
## `*` is elementwise (numpy's rule), so matrix multiplication is `matmul`,
## with `dot` covering numpy's four shape cases. Everything here works on
## `float` arrays; integer matrices go through `astype(float)` first, which
## is also where an ill-conditioned solve would want to be anyway.

import std/[math, fenv]
import ./core
import ./shape
import ./creation
import ./slicing
import ./index

proc dot*[T](a, b: NDArray[T]): T =
  ## Inner product of two 1-d arrays of the same length.
  if a.ndim != 1 or b.ndim != 1:
    raise newException(ValueError, "dot(1-d, 1-d) expects two vectors")
  if a.size != b.size:
    raise newException(ValueError,
      "dot: lengths " & $a.size & " and " & $b.size & " differ")
  result = T(0)
  for oa, ob in offsets2(a, b):
    result = result + a.buf[oa] * b.buf[ob]

proc matmul*[T](a, b: NDArray[T]): NDArray[T] =
  ## Matrix product. 2-d x 2-d, 2-d x 1-d and 1-d x 2-d all work; the 1-d
  ## operand is treated as a row or column as position demands and the added
  ## axis is dropped again, exactly as numpy does.
  if a.ndim notin 1..2 or b.ndim notin 1..2:
    raise newException(ValueError, "matmul takes 1-d or 2-d arrays")
  if a.ndim == 1 and b.ndim == 1:
    return initNDArray[T](@[], @[dot(a, b)])
  let am = if a.ndim == 1: a.reshape(1, a.size) else: a
  let bm = if b.ndim == 1: b.reshape(b.size, 1) else: b
  if am.shape[1] != bm.shape[0]:
    raise newException(ValueError,
      "matmul: shapes " & $a.shape & " and " & $b.shape & " do not line up")
  let n = am.shape[0]
  let k = am.shape[1]
  let m = bm.shape[1]
  var res = newNDArray[T](n, m)
  # read both operands into contiguous buffers first: the inner loop then
  # walks memory in order instead of chasing strides
  let ac = am.copy()
  let bc = bm.copy()
  for i in 0 ..< n:
    for p in 0 ..< k:
      let av = ac.buf[i * k + p]
      if av != T(0):
        let brow = p * m
        let crow = i * m
        for j in 0 ..< m:
          res.buf[crow + j] = res.buf[crow + j] + av * bc.buf[brow + j]
  if a.ndim == 1 and b.ndim == 2: res.reshape(m)
  elif a.ndim == 2 and b.ndim == 1: res.reshape(n)
  else: res

proc outer*[T](a, b: NDArray[T]): NDArray[T] =
  ## Outer product of two vectors: `a` as a column times `b` as a row.
  matmul(a.reshape(a.size, 1), b.reshape(1, b.size))

proc trace*[T](a: NDArray[T]): T =
  if a.ndim != 2: raise newException(ValueError, "trace takes a 2-d array")
  result = T(0)
  for i in 0 ..< min(a.shape[0], a.shape[1]): result = result + a[i, i]

proc diag*[T](a: NDArray[T]): NDArray[T] =
  ## Both of numpy's meanings, chosen by rank: the diagonal of a matrix as a
  ## vector, or a vector as the diagonal of a new matrix.
  if a.ndim == 2:
    let n = min(a.shape[0], a.shape[1])
    result = newNDArray[T](n)
    for i in 0 ..< n: result.buf[i] = a[i, i]
  elif a.ndim == 1:
    result = newNDArray[T](a.size, a.size)
    let xs = a.toSeq()
    for i in 0 ..< a.size: result[i, i] = xs[i]
  else:
    raise newException(ValueError, "diag takes a 1-d or 2-d array")

proc norm*[T: SomeFloat](a: NDArray[T], p = 2.0): T =
  ## Vector p-norm over every element (the Frobenius norm for a matrix at
  ## `p = 2`). `p = Inf` is the largest absolute value.
  if p == Inf:
    result = T(0)
    for x in a: result = max(result, abs(x))
  elif p == 1.0:
    result = T(0)
    for x in a: result = result + abs(x)
  elif p == 2.0:
    var s = T(0)
    for x in a: s = s + x * x
    result = sqrt(s)
  else:
    var s = T(0)
    for x in a: s = s + pow(abs(x), T(p))
    result = pow(s, T(1.0 / p))

# ------------------------------------------------- LU-based solvers --------

type LU = object
  ## The factorisation `PA = LU`, both triangles packed into one matrix.
  ## `solve`, `inv` and `det` are all readings of the same decomposition, so
  ## it is computed once here rather than three times.
  m: NDArray[float]
  perm: seq[int]
  swaps: int
  singular: bool

proc luDecompose(a: NDArray[float]): LU =
  if a.ndim != 2 or a.shape[0] != a.shape[1]:
    raise newException(ValueError, "expected a square matrix, got " & $a.shape)
  let n = a.shape[0]
  result.m = a.copy()
  result.perm = newSeq[int](n)
  for i in 0 ..< n: result.perm[i] = i
  for k in 0 ..< n:
    # partial pivoting: the largest available pivot, for numerical stability
    var piv = k
    var best = abs(result.m[k, k])
    for i in k + 1 ..< n:
      let v = abs(result.m[i, k])
      if v > best:
        best = v
        piv = i
    if best == 0.0:
      result.singular = true
      continue
    if piv != k:
      for j in 0 ..< n:
        let tmp = result.m[k, j]
        result.m[k, j] = result.m[piv, j]
        result.m[piv, j] = tmp
      swap(result.perm[k], result.perm[piv])
      result.swaps.inc
    let pivot = result.m[k, k]
    for i in k + 1 ..< n:
      let f = result.m[i, k] / pivot
      result.m[i, k] = f
      for j in k + 1 ..< n:
        result.m[i, j] = result.m[i, j] - f * result.m[k, j]

proc det*(a: NDArray[float]): float =
  ## Determinant, from the LU factorisation: the product of the pivots, with
  ## a sign from the row swaps.
  let lu = luDecompose(a)
  if lu.singular: return 0.0
  result = if lu.swaps mod 2 == 0: 1.0 else: -1.0
  for i in 0 ..< a.shape[0]: result *= lu.m[i, i]

proc luSolveVec(lu: LU, b: seq[float]): seq[float] =
  let n = lu.perm.len
  var y = newSeq[float](n)
  for i in 0 ..< n:
    var s = b[lu.perm[i]]
    for j in 0 ..< i: s -= lu.m[i, j] * y[j]
    y[i] = s
  result = newSeq[float](n)
  for i in countdown(n - 1, 0):
    var s = y[i]
    for j in i + 1 ..< n: s -= lu.m[i, j] * result[j]
    result[i] = s / lu.m[i, i]

proc solve*(a: NDArray[float], b: NDArray[float]): NDArray[float] =
  ## Solve `A x = b` for a square `A`. `b` may be a vector or a matrix of
  ## several right-hand sides; the result has `b`'s shape.
  let lu = luDecompose(a)
  if lu.singular:
    raise newException(ValueError, "solve: matrix is singular")
  let n = a.shape[0]
  if b.ndim == 1:
    if b.size != n:
      raise newException(ValueError,
        "solve: A is " & $a.shape & " but b has length " & $b.size)
    return initNDArray[float](@[n], luSolveVec(lu, b.toSeq()))
  if b.ndim != 2 or b.shape[0] != n:
    raise newException(ValueError,
      "solve: A is " & $a.shape & " but B is " & $b.shape)
  let m = b.shape[1]
  result = newNDArray[float](n, m)
  for j in 0 ..< m:
    let x = luSolveVec(lu, b.select(All, j).toSeq())
    for i in 0 ..< n: result[i, j] = x[i]

proc inv*(a: NDArray[float]): NDArray[float] =
  ## Matrix inverse, as the solve against the identity. Prefer `solve` when
  ## the inverse is only on its way to a product — it is both faster and
  ## better conditioned.
  solve(a, eye[float](a.shape[0]))

# ------------------------------------------------- QR-based solvers --------

type QR = object
  ## The Householder factorisation `A = QR` of a tall `A`. `R` sits in the
  ## upper triangle of `m`; below it are the reflector vectors, scaled so
  ## their leading entry is an implicit 1, with their factors in `tau`.
  ## `Q` is never formed — `lstsq` only ever needs to apply it.
  m: NDArray[float]
  tau: seq[float]
  rankDeficient: bool

proc qrDecompose(a: NDArray[float]): QR =
  ## Householder QR. Reflecting `A` onto the axes one column at a time is
  ## backward stable, where forming `AᵀA` squares the condition number and
  ## throws away half the available precision before the solve begins.
  let rows = a.shape[0]
  let cols = a.shape[1]
  result.m = a.copy()
  result.tau = newSeq[float](cols)
  for k in 0 ..< cols:
    # the reflector that maps column k, below the diagonal, onto e1
    var nrm = 0.0
    for i in k ..< rows: nrm += result.m[i, k] * result.m[i, k]
    nrm = sqrt(nrm)
    if nrm == 0.0: continue        # a zero column: the pivot scan below catches it
    let xk = result.m[k, k]
    # the sign that moves xk away from zero: subtracting nearly equal
    # numbers here is the one cancellation this algorithm could suffer
    let beta = if xk >= 0.0: -nrm else: nrm
    result.tau[k] = (beta - xk) / beta
    let denom = xk - beta
    for i in k + 1 ..< rows: result.m[i, k] = result.m[i, k] / denom
    result.m[k, k] = beta
    for j in k + 1 ..< cols:
      var s = result.m[k, j]
      for i in k + 1 ..< rows: s += result.m[i, k] * result.m[i, j]
      s *= result.tau[k]
      result.m[k, j] = result.m[k, j] - s
      for i in k + 1 ..< rows:
        result.m[i, j] = result.m[i, j] - s * result.m[i, k]
  # Rank comes from `R`'s diagonal, but a dependent column leaves rounding
  # noise there rather than an exact zero — two equal columns give a pivot
  # of 1e-16, not 0 — so the cutoff is relative to the largest pivot, as
  # LAPACK's rcond is. Dividing by the noise instead would return a huge
  # answer with no digits in it.
  var maxPivot = 0.0
  for k in 0 ..< cols: maxPivot = max(maxPivot, abs(result.m[k, k]))
  let tol = float(max(rows, cols)) * epsilon(float) * maxPivot
  for k in 0 ..< cols:
    if abs(result.m[k, k]) <= tol: result.rankDeficient = true

proc qrSolveVec(qr: QR, rows, cols: int, b: seq[float]): seq[float] =
  ## `Qᵀb` by replaying the reflectors, then back-substitution through `R`.
  var y = b
  for k in 0 ..< cols:
    if qr.tau[k] == 0.0: continue
    var s = y[k]
    for i in k + 1 ..< rows: s += qr.m[i, k] * y[i]
    s *= qr.tau[k]
    y[k] = y[k] - s
    for i in k + 1 ..< rows: y[i] = y[i] - s * qr.m[i, k]
  result = newSeq[float](cols)
  for i in countdown(cols - 1, 0):
    var s = y[i]
    for j in i + 1 ..< cols: s -= qr.m[i, j] * result[j]
    result[i] = s / qr.m[i, i]

proc lstsq*(a, b: NDArray[float]): NDArray[float] =
  ## Least-squares solution of an overdetermined `A x = b`, by Householder
  ## QR. `b` may be a vector or a matrix of several right-hand sides; the
  ## result has as many rows as `A` has columns.
  ##
  ## `A` must have full column rank and at least as many rows as columns —
  ## an underdetermined system has a space of solutions rather than one, and
  ## picking the shortest of them wants an SVD this library does not have.
  if a.ndim != 2:
    raise newException(ValueError, "lstsq: A must be 2-d, got " & $a.shape)
  let rows = a.shape[0]
  let cols = a.shape[1]
  if rows < cols:
    raise newException(ValueError,
      "lstsq: A is " & $a.shape & " — underdetermined systems need an SVD")
  if b.shape.len == 0 or b.shape[0] != rows:
    raise newException(ValueError,
      "lstsq: A is " & $a.shape & " but b is " & $b.shape)
  let qr = qrDecompose(a)
  if qr.rankDeficient:
    raise newException(ValueError, "lstsq: A is rank-deficient")
  if b.ndim == 1:
    return initNDArray[float](@[cols], qrSolveVec(qr, rows, cols, b.toSeq()))
  if b.ndim != 2:
    raise newException(ValueError, "lstsq: b must be 1-d or 2-d, got " & $b.shape)
  let k = b.shape[1]
  result = newNDArray[float](cols, k)
  for j in 0 ..< k:
    let x = qrSolveVec(qr, rows, cols, b.select(All, j).toSeq())
    for i in 0 ..< cols: result[i, j] = x[i]
