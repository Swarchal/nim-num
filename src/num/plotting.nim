## Bridge to [plotter](https://github.com/Swarchal/nim-plotter) charts.
##
## Not imported by `num` — plotter is not a dependency of this package and
## Nim has no optional ones, so importing this module is what pulls it in.
## It is `num/plotting`, not `num/plotter`: a module cannot import another of
## its own name, and this one has to reach plotter's umbrella module for the
## convenience constructors.
##
## ```nim
## import num
## import num/plotting         # adds scatter/line/hist over NDArray
## import plotter
##
## let x = linspace(0.0, 6.28, 100)
## scatter(x, sin(x)).save("sine.png")
## ```
##
## These are the same convenience constructors plotter already exposes, over
## arrays instead of `seq[float]`, plus `histChart`, which bins here rather
## than in Vega-Lite so the bins are the ones `histogram` computed.
##
## Arrays are flattened to 1-d on the way in: a chart axis is a vector, and
## an accidental 2-d argument almost always means a row or column was meant,
## which `a.row(i)` says explicitly.

import ./core
import ./shape
import ./index
import ./slicing
import ./stats
import plotter as pl
import plotter/data as pdata

proc toFloatSeq*(a: NDArray[float]): seq[float] =
  ## The values as a flat `seq[float]` — what every plotter entry point
  ## takes. A view is copied, since plotter owns its data.
  a.toSeq()

proc toDataColumn*(name: string, a: NDArray[float]): pdata.DataColumn =
  ## One array as one named plotter column.
  pdata.col(name, a.toSeq())

proc toDataFrame*(named: openArray[(string, NDArray[float])]): pdata.DataFrame =
  ## Parallel arrays as a tidy frame, one column each. Every array must be
  ## the same length — a frame's columns are rows of one table.
  var n = -1
  for (name, a) in named:
    if n < 0: n = a.size
    elif a.size != n:
      raise newException(ValueError,
        "column '" & name & "' has " & $a.size & " values, expected " & $n)
    result.columns.add(toDataColumn(name, a))

proc toDataFrame*(a: NDArray[float], names: openArray[string] = []): pdata.DataFrame =
  ## A 2-d array as a frame, one column per array column; names default to
  ## `c0`, `c1`, ...
  if a.ndim != 2:
    raise newException(ValueError, "toDataFrame takes a 2-d array, got " & $a.shape)
  for j in 0 ..< a.shape[1]:
    let name = if j < names.len: names[j] else: "c" & $j
    result.columns.add(toDataColumn(name, a[All, j].copy()))

proc toSeries*(name: string, xs, ys: NDArray[float]): pdata.Series =
  ## A named x/y pair, for plotter's multi-series overloads.
  if xs.size != ys.size:
    raise newException(ValueError,
      "series '" & name & "': " & $xs.size & " xs and " & $ys.size & " ys")
  pdata.Series(name: name, xs: xs.toSeq(), ys: ys.toSeq())

proc scatter*(xs, ys: NDArray[float], title = ""): pl.Chart =
  pl.scatter(xs.toSeq(), ys.toSeq(), title)

proc line*(xs, ys: NDArray[float], title = ""): pl.Chart =
  pl.line(xs.toSeq(), ys.toSeq(), title)

proc bar*(xs, ys: NDArray[float], title = ""): pl.Chart =
  pl.bar(xs.toSeq(), ys.toSeq(), title)

proc hist*(xs: NDArray[float], title = "", maxbins = 20): pl.Chart =
  ## Plotter's own histogram: the binning happens in Vega-Lite.
  pl.hist(xs.toSeq(), title, maxbins)

proc histChart*(xs: NDArray[float], bins = 10, title = ""): pl.Chart =
  ## A histogram binned **here**, by `num.histogram`, and drawn as bars from
  ## the bin centres. Use it when the counts have to match what the rest of
  ## an analysis computed; `hist` is the one to use when they need not.
  let (counts, edges) = histogram(xs, bins)
  var centres = newSeq[float](bins)
  for i in 0 ..< bins:
    centres[i] = (edges[i] + edges[i + 1]) / 2.0
  pl.bar(centres, counts.astype(float).toSeq(), title)

proc heatmap*(a: NDArray[float], title = "",
              rowNames: openArray[string] = [],
              colNames: openArray[string] = []): pl.Chart =
  ## A 2-d array as a heatmap. Row and column labels default to their
  ## indices as strings, since plotter's heatmap addresses cells by name.
  if a.ndim != 2:
    raise newException(ValueError, "heatmap takes a 2-d array, got " & $a.shape)
  var xs, ys: seq[string]
  var vals: seq[float]
  for i in 0 ..< a.shape[0]:
    for j in 0 ..< a.shape[1]:
      ys.add(if i < rowNames.len: rowNames[i] else: $i)
      xs.add(if j < colNames.len: colNames[j] else: $j)
      vals.add(a[i, j])
  pl.heatmap(xs, ys, vals, title)
