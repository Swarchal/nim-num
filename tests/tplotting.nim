## The plotter bridge. Plotter is not a dependency, so the suite is behind
## `when defined(plotter)` and `nimble test` still compiles this file with
## plotter absent. `nimble plottertest` runs it for real
## (`PLOTTER_PATH=/path/to/nim-plotter/src nimble plottertest`).

when defined(plotter):
  import std/[unittest, json]
  import num
  import num/plotting
  import plotter

  suite "arrays into charts":
    let x = linspace(0.0, 1.0, 5)

    test "scatter and line take arrays":
      let spec = scatter(x, x * 2.0, "t").toSpec()
      check spec["mark"].getStr == "point"
      check spec["data"]["values"].len == 5
      check line(x, x).toSpec()["mark"].getStr == "line"

    test "a chart is built from the values, not the layout":
      let a = arange(6).reshape(2, 3).astype(float)
      let spec = scatter(a.row(0), a.row(1)).toSpec()
      check spec["data"]["values"].len == 3

    test "histChart bins here, so the bars match num.histogram":
      let (counts, _) = histogram(x, 2)
      let spec = histChart(x, bins = 2).toSpec()
      check spec["data"]["values"].len == 2
      check spec["data"]["values"][0]["y"].getFloat == float(counts[0])

    test "hist leaves the binning to Vega-Lite":
      check hist(x).toSpec()["encoding"]["x"]["bin"].hasKey("maxbins")

  suite "arrays into frames":
    test "named arrays become a tidy frame":
      let df = toDataFrame({"x": arange(3).astype(float),
                            "y": ones[float](3)})
      check df.columns.len == 2
      check df.columns[0].name == "x"
      check df.columns[0].values.len == 3
      expect ValueError:
        discard toDataFrame({"x": ones[float](3), "y": ones[float](2)})

    test "a 2-d array becomes a frame, one column per column":
      let df = toDataFrame(arange(6).reshape(3, 2).astype(float), @["a", "b"])
      check df.columns.len == 2
      check df.columns[1].values.len == 3
      check toDataFrame(zeros[float](2, 2)).columns[0].name == "c0"

    test "toSeries pairs two arrays":
      let s = toSeries("run 1", arange(3).astype(float), ones[float](3))
      check s.name == "run 1"
      check s.xs.len == 3
      expect ValueError:
        discard toSeries("bad", ones[float](3), ones[float](2))

    test "heatmap labels default to indices":
      let spec = heatmap(arange(4).reshape(2, 2).astype(float)).toSpec()
      check spec["data"]["values"].len == 4
