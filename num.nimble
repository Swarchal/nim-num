# Package

version       = "0.1.0"
author        = "Scott Warchal"
description   = "N-dimensional numeric arrays for Nim, in the spirit of numpy"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"

task racoontest, "Run the racoon bridge test (racoon is not a dependency)":
  # `nimble test` compiles tests/tracoon.nim with the suite switched off, so
  # racoon need not be present for the normal run. Set RACOON_PATH to a
  # checkout's src/ if racoon is not installed as a package.
  let racoonPath = getEnv("RACOON_PATH")
  let extra = if racoonPath.len > 0: " --path:" & racoonPath else: ""
  exec "nim c -r --path:src -d:racoon --hints:off" & extra & " tests/tracoon.nim"

task plottertest, "Run the plotter bridge test (plotter is not a dependency)":
  let plotterPath = getEnv("PLOTTER_PATH")
  let extra = if plotterPath.len > 0: " --path:" & plotterPath else: ""
  exec "nim c -r --path:src -d:plotter --hints:off" & extra & " tests/tplotting.nim"
