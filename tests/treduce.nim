import std/[unittest, math]
import num

suite "whole-array folds":
  let a = arange(6).reshape(2, 3)          # [[0,1,2],[3,4,5]]

  test "sum, prod, min, max":
    check sum(a) == 15
    check prod(arange(1, 5)) == 24
    check min(a) == 0
    check max(a) == 5
    check ptp(a) == 5

  test "argmin and argmax are flat indices":
    check argmax(a) == 5
    check argmin(a) == 0
    check argmax(toNDArray(@[1, 3, 3])) == 1     # ties go to the first

  test "mean, variance, std":
    check mean(a) == 2.5
    check mean(arange(4)) == 1.5
    check variance(toNDArray(@[1.0, 2.0, 3.0])) == 2.0 / 3.0
    check variance(toNDArray(@[1.0, 2.0, 3.0]), ddof = 1) == 1.0
    check abs(std(toNDArray(@[1.0, 2.0, 3.0]), ddof = 1) - 1.0) < 1e-12

  test "empty folds raise rather than invent an identity":
    expect ValueError: discard max(zeros[int](0))
    expect ValueError: discard mean(zeros[int](0))
    check sum(zeros[int](0)) == 0              # ... but a sum has one

  test "booleans":
    check all(toNDArray(@[true, true]))
    check not all(toNDArray(@[true, false]))
    check any(toNDArray(@[false, true]))
    check countNonZero(toNDArray(@[0, 1, 2])) == 2
    check countNonZero(toNDArray(@[true, false, true])) == 2
    check sum(arange(6).reshape(2, 3) > 2) == 3      # a mask sums to a count

suite "axis folds":
  let a = arange(6).reshape(2, 3)

  test "the axis is removed":
    check sum(a, axis = 0).shape == @[3]
    check sum(a, axis = 0).toSeq() == @[3, 5, 7]
    check sum(a, axis = 1).toSeq() == @[3, 12]
    check sum(a, axis = -1).toSeq() == @[3, 12]
    expect ValueError: discard sum(a, axis = 2)

  test "keepDims leaves it at length 1, so the result broadcasts back":
    let m = mean(a, axis = 1, keepDims = true)
    check m.shape == @[2, 1]
    let centred = a.astype(float) - m
    check centred.toSeq() == @[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]

  test "min, max, mean, std along an axis":
    check min(a, axis = 0).toSeq() == @[0, 1, 2]
    check max(a, axis = 1).toSeq() == @[2, 5]
    check mean(a, axis = 0).toSeq() == @[1.5, 2.5, 3.5]
    check allclose(std(a, axis = 1), toNDArray(@[sqrt(2.0 / 3.0), sqrt(2.0 / 3.0)]))
    check allclose(variance(a, axis = 1, ddof = 1), toNDArray(@[1.0, 1.0]))

  test "argmin/argmax give positions along the axis":
    let b = toNDArray(@[@[3, 1, 2], @[0, 9, 4]])
    check argmax(b, axis = 1).toSeq() == @[0, 1]
    check argmin(b, axis = 0).toSeq() == @[1, 0, 0]

  test "summing a mask counts":
    check sum(a > 2, axis = 1).toSeq() == @[0, 3]
    check all(a >= 0, axis = 0).toSeq() == @[true, true, true]
    check any(a > 4, axis = 0).toSeq() == @[false, false, true]

suite "scans":
  test "cumsum and cumprod flatten by default":
    check cumsum(arange(4)).toSeq() == @[0, 1, 3, 6]
    check cumprod(arange(1, 5)).toSeq() == @[1, 2, 6, 24]

  test "cumsum along an axis keeps the shape":
    let a = arange(6).reshape(2, 3)
    check cumsum(a, axis = 1).toSeq() == @[0, 1, 3, 3, 7, 12]
    check cumsum(a, axis = 0).toSeq() == @[0, 1, 2, 3, 5, 7]

  test "diff":
    check diff(toNDArray(@[1, 4, 9])).toSeq() == @[3, 5]
    check diff(zeros[int](1)).size == 0

suite "NaN-aware folds":
  # the shape a racoon column with holes arrives in: the bridge maps NA to
  # NaN, and these are what still give an answer over it
  let a = toNDArray(@[@[1.0, NaN, 3.0], @[NaN, NaN, 6.0]])

  test "the plain folds still propagate NaN":
    check isNaN(sum(a))
    check isNaN(mean(a))

  test "whole-array nan folds skip the holes":
    check nanCount(a) == 3
    check nanSum(a) == 10.0
    check abs(nanMean(a) - 3.3333333333333335) < 1e-12
    check nanMin(a) == 1.0
    check nanMax(a) == 6.0
    check abs(nanVariance(a) - 4.222222222222222) < 1e-12
    check abs(nanStd(a) - 2.0548046676563256) < 1e-12
    check abs(nanVariance(a, ddof = 1) - 6.333333333333333) < 1e-12

  test "numpy's spelling works — Nim reads nansum and nanSum as one name":
    check nansum(a) == 10.0
    check nanmax(a) == 6.0

  test "along an axis, an all-NaN slice gives NaN and spares the others":
    check nanSum(a, axis = 0).toSeq() == @[1.0, 0.0, 9.0]
    check nanCount(a, axis = 0).toSeq() == @[1, 0, 2]
    let m = nanMean(a, axis = 0)
    check m[0] == 1.0
    check isNaN(m[1])
    check m[2] == 4.5
    let lo = nanMin(a, axis = 0)
    check lo[0] == 1.0
    check isNaN(lo[1])
    check lo[2] == 3.0
    let hi = nanMax(a, axis = 0)
    check hi[2] == 6.0
    check isNaN(hi[1])

  test "the other axis, and the moments along it":
    check nanSum(a, axis = 1).toSeq() == @[4.0, 6.0]
    check nanMean(a, axis = 1).toSeq() == @[2.0, 6.0]
    check nanVariance(a, axis = 1).toSeq() == @[1.0, 0.0]
    check nanStd(a, axis = 1).toSeq() == @[1.0, 0.0]
    # one real value is not enough for a sample variance: that slice alone
    # goes NaN rather than taking the whole call down with it
    let v = nanVariance(a, axis = 1, ddof = 1)
    check v[0] == 2.0
    check isNaN(v[1])

  test "an all-NaN whole-array fold raises where there is no answer":
    let allNaN = full(@[3], NaN)
    check nanSum(allNaN) == 0.0            # 0 is the identity, as for sum
    check nanCount(allNaN) == 0
    expect ValueError: discard nanMean(allNaN)
    expect ValueError: discard nanMin(allNaN)
    expect ValueError: discard nanMax(allNaN)
    expect ValueError: discard nanVariance(allNaN)

  test "an array with no NaN in it folds exactly as the plain form does":
    let clean = arange(6).astype(float).reshape(2, 3)
    check nanSum(clean) == sum(clean)
    check nanMean(clean) == mean(clean)
    check nanMin(clean, axis = 1).toSeq() == min(clean, axis = 1).toSeq()
    check nanVariance(clean, axis = 0).toSeq() == variance(clean, axis = 0).toSeq()

  test "Inf is a value, not a hole":
    let withInf = toNDArray(@[1.0, Inf, NaN])
    check nanMax(withInf) == Inf
    check nanCount(withInf) == 2
    # an all-Inf slice must not be mistaken for an empty one
    check nanMax(toNDArray(@[@[Inf, Inf]]), axis = 1).toSeq() == @[Inf]
