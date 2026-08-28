import std/unittest
import num

suite "random arrays":
  setup:
    seed(42)

  test "shapes and ranges":
    let a = randArray(2, 3)
    check a.shape == @[2, 3]
    check min(a) >= 0.0 and max(a) < 1.0
    let u = uniform(-1.0, 1.0, 100)
    check min(u) >= -1.0 and max(u) < 1.0
    let i = randint(0, 5, 50)
    check min(i) >= 0 and max(i) < 5
    expect ValueError: discard randint(5, 5, 2)

  test "seeding makes a run reproducible":
    seed(7)
    let a = randArray(5)
    seed(7)
    check randArray(5) == a

  test "randn is roughly standard normal":
    let a = randn(20000)
    check abs(mean(a)) < 0.05
    check abs(std(a) - 1.0) < 0.05
    check a.size == 20000              # the odd tail of a Box-Muller pair

  test "choice and shuffled keep the values":
    let a = arange(5)
    check choice(a, 3, replace = false).size == 3
    check sortedArray(shuffled(a)) == a
    check sortedArray(choice(a, 5, replace = false)) == a
    expect ValueError: discard choice(a, 6, replace = false)

  test "permutation is a permutation":
    check sortedArray(permutation(10)) == arange(10)
