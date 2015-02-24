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

print("a[i,j] = 2 * b[i,j]")
a["i,j"] = 2 * b["i,j"]
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

c = ambit.Tensor(ambit.TensorType.kCore, "C", [5, 5])
ambit.initialize_random(c.tensor)

print("a[i,j] = b[i,k] * c[k,j]")
a["i,j"] = b["i,k"] * c["k,j"]
a.printf()

print("a[i,j] = b[i,k] * c[k,j] * 2")
a["i,j"] = b["i,k"] * c["k,j"] * 2
a.printf()

print("a[i,j] = 2 * b[i,k] * c[k,j]")
a["i,j"] = 2 * b["i,k"] * c["k,j"]
a.printf()
