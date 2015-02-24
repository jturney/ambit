import ambit
a = ambit.Tensor(ambit.TensorType.kCore, "A", [5, 5])
b = ambit.Tensor(ambit.TensorType.kCore, "B", [5, 5])

ambit.initialize_random(a.tensor)
ambit.initialize_random(b.tensor)

a.printf()
b.printf()

print("a[i,j] = b[i,j]")
a["i,j"] = b["i,j"]
a.printf()

print("a[i,j] += b[i,j]")
a["i,j"] += b["i,j"]
a.printf()

print("a[i,j] -= b[i,j]")
a["i,j"] -= b["i,j"]
a.printf()

print("a[i,j] = b[i,j] + b[i,j]")
a["i,j"] = b["i,j"] + b["i,j"]
a.printf()

print("a[i,j] = b[i,j] - b[i,j]")
a["i,j"] = b["i,j"] - b["i,j"]
a.printf()
