## The racoon bridge. Racoon is not a dependency, so the suite is behind
## `when defined(racoon)` and `nimble test` still compiles this file with
## racoon absent. `nimble racoontest` runs it for real
## (`RACOON_PATH=/path/to/Racoon/src nimble racoontest`).

when defined(racoon):
  import std/[unittest, math, options]
  import num
  import num/racoon
  import racoon

  suite "racoon -> num":
    let df = toDataFrame("a,b,c\n1,2.5,x\n3,4.5,y\n")

    test "a numeric column becomes a 1-d float array":
      let a = df["a"].toArray()
      check a.shape == @[2]
      check a.toSeq() == @[1.0, 3.0]          # int widens to float
      check df["b"].toArray().toSeq() == @[2.5, 4.5]

    test "a string column has no numeric reading":
      expect ValueError: discard df["c"].toArray()

    test "NA becomes NaN in a float array":
      let na = toDataFrame("a\n1.0\n\n3.0\n")
      let a = na["a"].toArray()
      check a.isNaN.toSeq() == @[false, true, false]
      check a.size == 3

    test "an int array has no NaN, so NA raises unless filled":
      let na = toDataFrame("a\n1\n\n3\n")
      expect ValueError: discard na["a"].toIntArray()
      check na["a"].toIntArray(fill = some(0)).toSeq() == @[1, 0, 3]

    test "toMatrix builds a row-major design matrix":
      let m = df.toMatrix(@["a", "b"])
      check m.shape == @[2, 2]
      check m.toSeq() == @[1.0, 2.5, 3.0, 4.5]
      check m.row(0).toSeq() == @[1.0, 2.5]

    test "the sweep form skips non-numeric columns":
      check df.toMatrix().shape == @[2, 2]
      expect ValueError: discard toDataFrame("c\nx\n").toMatrix()

  suite "num -> racoon":
    test "an array becomes a column":
      let c = toColumn("v", toNDArray(@[1.0, 2.0]))
      check c.name == "v"
      check c.kind == ckFloat
      check c.countNA == 0

    test "NaN stays a value unless it is said to be missing":
      let a = toNDArray(@[1.0, NaN])
      check toColumn("v", a).countNA == 0
      check toColumn("v", a, nanAsNA = true).countNA == 1

    test "a 2-d array becomes a frame":
      let f = toDataFrame(arange(6).reshape(3, 2).astype(float), @["x", "y"])
      check f.shape == [3, 2]
      check f.header == @["x", "y"]
      check f["y"].toArray().toSeq() == @[1.0, 3.0, 5.0]
      check toDataFrame(zeros[float](2, 2)).header == @["c0", "c1"]

    test "a round trip through racoon preserves the values":
      let m = arange(6).reshape(3, 2).astype(float)
      check toDataFrame(m, @["x", "y"]).toMatrix(@["x", "y"]) == m

    test "only 1-d arrays are columns":
      expect ValueError: discard toColumn("v", zeros[float](2, 2))
