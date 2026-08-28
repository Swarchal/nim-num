## Random arrays.
##
## These are `proc`, not `func`: they use `std/random`'s global RNG, so they
## have an effect. `seed` makes a run reproducible — the same split racoon
## makes for `sample`/`shuffle`.

import std/[random, math]
import ./core

proc seed*(x: int64) =
  ## Reseed the global RNG, for a reproducible run.
  randomize(x)

proc randomSeed*() =
  ## Reseed from the clock — the default state is *fixed*, so a program that
  ## wants different numbers on each run has to ask.
  randomize()

proc randArray*(shape: varargs[int]): NDArray[float] =
  ## Uniform on `[0, 1)`, numpy's `random.rand`.
  result = newNDArray[float](shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = rand(1.0)

proc uniform*(lo, hi: float, shape: varargs[int]): NDArray[float] =
  ## Uniform on `[lo, hi)`.
  result = newNDArray[float](shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = lo + rand(hi - lo)

proc randn*(shape: varargs[int]): NDArray[float] =
  ## Standard normal, by the Box-Muller transform. Both values of each pair
  ## are kept, so `n` draws cost `n/2` transforms.
  result = newNDArray[float](shape)
  var i = 0
  let n = result.buf[].len
  while i < n:
    let u1 = max(rand(1.0), 1e-12) # log(0) is not a number we can use
    let u2 = rand(1.0)
    let r = sqrt(-2.0 * ln(u1))
    result.buf[i] = r * cos(2.0 * PI * u2)
    if i + 1 < n: result.buf[i + 1] = r * sin(2.0 * PI * u2)
    i += 2

proc normal*(mu, sigma: float, shape: varargs[int]): NDArray[float] =
  result = randn(shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = mu + sigma * result.buf[i]

proc randint*(lo, hi: int, shape: varargs[int]): NDArray[int] =
  ## Uniform integers on the half-open `[lo, hi)`, as numpy's `randint` is.
  if hi <= lo: raise newException(ValueError, "randint: need hi > lo")
  result = newNDArray[int](shape)
  for i in 0 ..< result.buf[].len: result.buf[i] = lo + rand(hi - lo - 1)

proc choice*[T](a: NDArray[T], n: int, replace = true): NDArray[T] =
  ## `n` elements drawn from `a`'s values.
  let xs = a.toSeq()
  if xs.len == 0: raise newException(ValueError, "choice from an empty array")
  var picked = newSeq[T](n)
  if replace:
    for i in 0 ..< n: picked[i] = xs[rand(xs.high)]
  else:
    if n > xs.len:
      raise newException(ValueError,
        "choice: cannot draw " & $n & " of " & $xs.len & " without replacement")
    var pool = xs
    for i in 0 ..< n:
      let j = rand(pool.high)
      picked[i] = pool[j]
      pool.del(j)
  initNDArray[T](@[n], picked)

proc shuffled*[T](a: NDArray[T]): NDArray[T] =
  ## A new 1-d array holding `a`'s values in random order.
  var xs = a.toSeq()
  shuffle(xs)
  initNDArray[T](@[xs.len], xs)

proc permutation*(n: int): NDArray[int] =
  ## A random permutation of `0 ..< n` — the indices `take` wants.
  var idx = newSeq[int](n)
  for i in 0 ..< n: idx[i] = i
  shuffle(idx)
  initNDArray[int](@[n], idx)
