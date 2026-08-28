import std/[unittest, math]
import num

suite "order statistics":
  test "sortedArray and argsort":
    let a = toNDArray(@[3, 1, 2])
    check sortedArray(a).toSeq() == @[1, 2, 3]
    check sortedArray(a, ascending = false).toSeq() == @[3, 2, 1]
    check argsort(a).toSeq() == @[1, 2, 0]
    check a.take(argsort(a).toSeq()).toSeq() == @[1, 2, 3]

  test "argsort is stable in both directions":
    let a = toNDArray(@[1, 1, 0])
    check argsort(a).toSeq() == @[2, 0, 1]
    check argsort(a, ascending = false).toSeq() == @[0, 1, 2]

  test "median averages the two middles":
    check median(toNDArray(@[3, 1, 2])) == 2.0
    check median(toNDArray(@[4, 1, 2, 3])) == 2.5
    expect ValueError: discard median(zeros[int](0))

  test "quantile interpolates, and agrees with median at 0.5":
    let a = toNDArray(@[0.0, 1.0, 2.0, 3.0])
    check quantile(a, 0.0) == 0.0
    check quantile(a, 1.0) == 3.0
    check quantile(a, 0.5) == median(a)
    check abs(quantile(a, 0.25) - 0.75) < 1e-12
    check percentile(a, 50.0) == 1.5
    check iqr(a) == 1.5
    expect ValueError: discard quantile(a, 1.5)

  test "unique is sorted and distinct":
    check unique(toNDArray(@[3, 1, 3, 1, 2])).toSeq() == @[1, 2, 3]

suite "order statistics with NaN in them":
  # A racoon column with holes arrives here as NaN. Nim's `cmp` is not a
  # valid comparator for floats, so before `nanLast` the answer depended on
  # where in the input the hole sat: these two used to give `nan` and `4.0`.
  let a = toNDArray(@[5.0, NaN, 1.0, 3.0])
  let b = toNDArray(@[NaN, 5.0, 1.0, 3.0])

  test "the ordering does not depend on where the hole was":
    check sortedArray(a).toSeq()[0 .. 2] == @[1.0, 3.0, 5.0]
    check sortedArray(b).toSeq()[0 .. 2] == @[1.0, 3.0, 5.0]
    check isNaN(sortedArray(a).toSeq()[3])
    check isNaN(sortedArray(b).toSeq()[3])

  test "NaN sorts last, and descending reverses the whole ordering":
    check isNaN(sortedArray(a, ascending = false).toSeq()[0])
    check sortedArray(a, ascending = false).toSeq()[1 .. 3] == @[5.0, 3.0, 1.0]
    check argsort(a).toSeq() == @[2, 3, 0, 1]      # the hole's index goes last

  test "the order statistics propagate":
    check isNaN(median(a))
    check isNaN(median(b))                          # the one that used to lie
    check isNaN(quantile(a, 0.25))
    check isNaN(percentile(a, 90.0))
    check isNaN(iqr(a))
    check isNaN(quantiles(a, @[0.0, 1.0])[0])

  test "the nan forms answer over what is there":
    check nanMedian(a) == 3.0
    check nanMedian(b) == 3.0
    check nanQuantile(a, 0.5) == median(toNDArray(@[1.0, 3.0, 5.0]))
    check nanQuantile(a, 0.0) == 1.0
    expect ValueError: discard nanMedian(full(@[3], NaN))
    expect ValueError: discard nanQuantile(full(@[3], NaN), 0.5)
    expect ValueError: discard nanQuantile(a, 1.5)

  test "a clean array is untouched by any of it":
    let clean = toNDArray(@[3.0, 1.0, 2.0])
    check sortedArray(clean).toSeq() == @[1.0, 2.0, 3.0]
    check nanMedian(clean) == median(clean)
    check nanQuantile(clean, 0.25) == quantile(clean, 0.25)
    check median(toNDArray(@[3, 1, 2])) == 2.0     # ints never see the branch

  test "unique keeps one NaN, at the end":
    let u = unique(toNDArray(@[3.0, NaN, 1.0, NaN, 3.0]))
    check u.size == 3
    check u.toSeq()[0 .. 1] == @[1.0, 3.0]
    check isNaN(u[2])
    check unique(full(@[4], NaN)).size == 1

suite "distributions of values":
  test "bincount":
    check bincount(toNDArray(@[0, 1, 1, 3])).toSeq() == @[1, 2, 0, 1]
    check bincount(toNDArray(@[0]), minLength = 3).size == 3
    expect ValueError: discard bincount(toNDArray(@[-1]))

  test "histogram counts every point exactly once":
    let a = toNDArray(@[0.0, 1.0, 2.0, 3.0, 4.0])
    let (counts, edges) = histogram(a, bins = 2)
    check counts.toSeq() == @[2, 3]        # the right edge is in the last bin
    check sum(counts) == a.size
    check edges.toSeq() == @[0.0, 2.0, 4.0]

  test "histogram of a constant array still has a bin":
    let (counts, _) = histogram(toNDArray(@[2.0, 2.0]), bins = 2)
    check sum(counts) == 2

  test "an explicit range drops what falls outside":
    let (counts, _) = histogram(toNDArray(@[0.0, 5.0, 10.0]), bins = 2,
                                range = (0.0, 6.0))
    check sum(counts) == 2

suite "two-variable summaries":
  let x = toNDArray(@[1.0, 2.0, 3.0, 4.0])

  test "cov and corr":
    check abs(cov(x, x) - variance(x, ddof = 1)) < 1e-12
    check abs(corr(x, x) - 1.0) < 1e-12
    check abs(corr(x, x * -1.0) + 1.0) < 1e-12
    expect ValueError: discard corr(x, ones[float](4))

  test "normalize and zscore":
    check normalize(x).toSeq() == @[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    check abs(mean(zscore(x))) < 1e-12
    check abs(std(zscore(x)) - 1.0) < 1e-12
    check normalize(ones[float](3)).toSeq() == @[0.0, 0.0, 0.0]
