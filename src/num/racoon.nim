## Bridge to [racoon](https://github.com/Swarchal/Racoon) DataFrames.
##
## Not imported by `num` — racoon is not a dependency of this package and Nim
## has no optional ones, so importing this module is what pulls it in:
##
## ```nim
## import num
## import num/racoon           # adds toArray / toColumn / toMatrix
## import racoon
##
## let df = readFile("iris.csv").toDataFrame()
## let x = df["sepal_length"].toArray()        # racoon Column -> NDArray
## echo x.mean()
## echo df.toMatrix(@["sepal_length", "petal_length"]).shape
## ```
##
## The two libraries can be imported unqualified: their procs differ in their
## parameter types. Only the bare type name `DataFrame` is unambiguous here
## (num has none), but `toDataFrame` exists in both racoon and plotter, so
## qualify it if all three are in scope.
##
## **Missing values.** Racoon's NA is a validity mask; an `NDArray` has no
## such thing, so the two directions are deliberately not symmetric:
##
## - `toArray` on a float column maps NA to **NaN** — the only float that
##   carries "no value" — and on an int/bool/string column *raises*, because
##   those types have no such spelling. Pass `fill = ...` to substitute a
##   value instead.
## - `toColumn` maps nothing back by default: racoon treats NaN as an
##   ordinary float value, not a missing one, and silently promoting it would
##   contradict that. Pass `nanAsNA = true` when the NaNs did come from NA.
##
## Cells are read through `col[i]`, never the payload seq — under a missing
## cell the payload holds the branch default.

import std/[math, options]
import racoon/value as rvalue
import racoon/dataframe as rdataframe
import ./core
import ./index

proc toArray*(c: rvalue.Column): NDArray[float] =
  ## A numeric racoon column as a 1-d float array. Int columns widen, which
  ## is racoon's own `commonKind` rule; NA becomes NaN.
  if c.kind notin {ckInt, ckFloat}:
    raise newException(ValueError,
      "column '" & c.name & "' is " & $c.kind & ", not numeric")
  result = newNDArray[float](c.len)
  for i in 0 ..< c.len:
    result.buf[i] = if c.isNA(i): NaN else: c.floatAt(i)

proc toIntArray*(c: rvalue.Column, fill = none(int)): NDArray[int] =
  ## An int column as an int array. A missing cell raises unless `fill`
  ## says what to put there — `int` has no NaN to fall back on.
  if c.kind != ckInt:
    raise newException(ValueError,
      "column '" & c.name & "' is " & $c.kind & ", not int")
  result = newNDArray[int](c.len)
  for i in 0 ..< c.len:
    if c.isNA(i):
      if fill.isNone:
        raise newException(ValueError,
          "column '" & c.name & "' has a missing value at row " & $i &
          "; pass fill = some(x) to substitute one")
      result.buf[i] = fill.get
    else:
      result.buf[i] = c[i].i

proc toBoolArray*(c: rvalue.Column, fill = none(bool)): NDArray[bool] =
  if c.kind != ckBool:
    raise newException(ValueError,
      "column '" & c.name & "' is " & $c.kind & ", not bool")
  result = newNDArray[bool](c.len)
  for i in 0 ..< c.len:
    if c.isNA(i):
      if fill.isNone:
        raise newException(ValueError,
          "column '" & c.name & "' has a missing value at row " & $i &
          "; pass fill = some(x) to substitute one")
      result.buf[i] = fill.get
    else:
      result.buf[i] = c[i].b

proc toMatrix*(df: rdataframe.DataFrame, columns: openArray[string]): NDArray[float] =
  ## The named numeric columns as a row-major `rows x columns` float matrix —
  ## the design matrix a fit or a distance wants. Column-major to row-major
  ## is a genuine transpose, so this copies.
  if columns.len == 0:
    raise newException(ValueError, "toMatrix needs at least one column")
  let cols = block:
    var acc: seq[NDArray[float]]
    for name in columns: acc.add(df[name].toArray())
    acc
  let rows = cols[0].size
  for i, c in cols:
    if c.size != rows:
      raise newException(ValueError, "ragged frame: column '" & columns[i] & "'")
  result = newNDArray[float](rows, columns.len)
  for j, c in cols:
    for i in 0 ..< rows:
      result[i, j] = c.buf[i]

proc toMatrix*(df: rdataframe.DataFrame): NDArray[float] =
  ## Every numeric column of the frame, in frame order. String and bool
  ## columns are **skipped** rather than raising — a frame is rarely all
  ## numeric, and this is the sweep form, following racoon's own rule that
  ## naming a column is a request but a sweep may skip.
  var names: seq[string]
  for c in df.columns:
    if c.kind in {ckInt, ckFloat}: names.add(c.name)
  if names.len == 0:
    raise newException(ValueError, "frame has no numeric columns")
  toMatrix(df, names)

proc toColumn*(name: string, a: NDArray[float],
    nanAsNA = false): rvalue.Column =
  ## A 1-d float array as a racoon column. NaN stays a value unless
  ## `nanAsNA` says it came from one.
  if a.ndim != 1:
    raise newException(ValueError, "toColumn takes a 1-d array, got " & $a.shape)
  result = rvalue.toColumn(name, a.toSeq())
  if nanAsNA:
    var valid = newSeq[bool](a.size)
    var i = 0
    for x in a:
      valid[i] = x == x
      i.inc
    result.valid = valid

proc toColumn*(name: string, a: NDArray[int]): rvalue.Column =
  if a.ndim != 1:
    raise newException(ValueError, "toColumn takes a 1-d array, got " & $a.shape)
  rvalue.toColumn(name, a.toSeq())

proc toColumn*(name: string, a: NDArray[bool]): rvalue.Column =
  if a.ndim != 1:
    raise newException(ValueError, "toColumn takes a 1-d array, got " & $a.shape)
  rvalue.toColumn(name, a.toSeq())

proc toDataFrame*(a: NDArray[float], names: openArray[string] = []): rdataframe.DataFrame =
  ## A 2-d array as a frame, one racoon column per array column. Names
  ## default to `c0`, `c1`, ... — racoon rejects duplicates, so a name list
  ## that repeats itself is caught there rather than here.
  if a.ndim != 2:
    raise newException(ValueError, "toDataFrame takes a 2-d array, got " & $a.shape)
  var cols: seq[rvalue.Column]
  for j in 0 ..< a.shape[1]:
    let name = if j < names.len: names[j] else: "c" & $j
    var vals = newSeq[float](a.shape[0])
    for i in 0 ..< a.shape[0]: vals[i] = a[i, j]
    cols.add(rvalue.toColumn(name, vals))
  rdataframe.toDataFrame(cols)
