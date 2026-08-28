## Pretty-printing beyond `$`: the one-line shape/dtype summary.

import std/strutils
import ./core

func dtype*[T](a: NDArray[T]): string =
  ## The element type's name, as a word — `int`, `float`, `bool`, `string`.
  ## A statement about the type, so it is here rather than at each caller.
  $T

proc describe*[T](a: NDArray[T]): string =
  ## `NDArray[float] shape=(2, 3) strides=(3, 1) contiguous` — what you want
  ## when the values are not the question.
  result = "NDArray[" & a.dtype & "] shape=(" & a.shape.join(", ") & ")"
  result.add(" strides=(" & a.strides.join(", ") & ")")
  if a.offset != 0: result.add(" offset=" & $a.offset)
  result.add(if a.isContiguous: " contiguous" else: " strided")
