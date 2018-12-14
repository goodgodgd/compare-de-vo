from collections import namedtuple

opt = namedtuple("options", "a1 a2 a3")
print("opt", opt)
foo = opt("foo", "bar", "baz")
print("foo", foo)
foo.far = 1
print("foo", foo)
