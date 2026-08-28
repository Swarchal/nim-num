## `[]` and `[]=`, dispatching between element access and slicing.
##
## `a[1, 2]` is an element and `a[1, All]` is a view, and the two cannot be
## plain overloads: `int` converts to `Sel`, so an all-integer call matches
## both signatures and Nim (rightly) calls it ambiguous. The dispatch is
## therefore made here, at the syntax level, by asking which underlying call
## compiles — `elemAt` for all-integer indices, `select` for anything with a
## `Sel`, a slice or a mask in it.
##
## The consequence to know: **an all-integer index must name every axis.**
## `a[1]` on a 2-d array is an error, not its first row; write `a[1, All]`
## or `a.row(1)`.
##
## `^k` is not an integer index — it converts to `Sel`, so it takes the
## `select` branch and `a[^1]` on a 1-d array is a 0-d array where `a[0]`
## is an element. `a[^1].item` is the element, and `a.toSeq()[^1]` reads
## the last value of any array.

import std/macros
import ./core
import ./slicing

template `[]`*[T](a: NDArray[T], args: varargs[untyped]): untyped =
  ## Element access when every index is an integer, a view otherwise. Typed
  ## on `NDArray[T]`, so it does not shadow `[]` for any other container.
  when compiles(elemAt(a, args)): elemAt(a, args)
  else: select(a, args)

macro `[]=`*[T](a: var NDArray[T], args: varargs[untyped]): untyped =
  ## The last argument is the value; everything before it is the index.
  ## A macro rather than a template because `varargs` cannot be split into
  ## "the indices" and "the value" in a signature.
  ##
  ## The array parameter is **typed**, so this catches only assignments into
  ## an `NDArray` — an untyped `[]=` macro would catch every
  ## assignment-through-brackets in every module that imports this one,
  ## `someSeq[i] = x` and `someRef[] = x` included. The helper names are
  ## bound with `bindSym` for the same reason: they resolve in this module's
  ## scope, not the caller's.
  let val = args[args.len - 1]
  var idx = newSeq[NimNode]()
  for i in 0 ..< args.len - 1: idx.add(args[i])
  let setElemSym = bindSym"setElem"
  let setSelectSym = bindSym"setSelect"
  let toSelSym = bindSym"toSel"
  let elemCall = newCall(setElemSym, a, nnkBracket.newTree(idx), val)
  var selArr = nnkBracket.newTree()
  for n in idx: selArr.add(newCall(toSelSym, n))
  let selCall = newCall(setSelectSym, a, selArr, val)
  if idx.len == 1:
    # a single non-integer index is a mask; `setSelect` takes it directly
    let maskCall = newCall(setSelectSym, a, idx[0], val)
    result = quote do:
      when compiles(`elemCall`): `elemCall`
      elif compiles(`maskCall`): `maskCall`
      else: `selCall`
  else:
    result = quote do:
      when compiles(`elemCall`): `elemCall`
      else: `selCall`
