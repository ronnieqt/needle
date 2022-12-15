#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import needle as ndl
ndl.autograd.LAZY_MODE = True

a = ndl.Tensor([1,2,3], requires_grad=False)
b = ndl.Tensor([4,5,6], requires_grad=False)
c = a + b
d = a + c
res = (c * d).sum()

print("=========================")
print(f"        a : {str(id(a))[-6:]}")
print(f"        b : {str(id(b))[-6:]}")
print(f"        c : {str(id(c))[-6:]}")
print(f"        d : {str(id(d))[-6:]}")
print(f"      res : {str(id(res))[-6:]}")
print(f"   a.grad : {str(id(a.grad))[-6:]}, {a.grad is not None}")
print(f"   b.grad : {str(id(b.grad))[-6:]}, {b.grad is not None}")
print(f"   c.grad : {str(id(c.grad))[-6:]}, {c.grad is not None}")
print(f"   d.grad : {str(id(d.grad))[-6:]}, {d.grad is not None}")
print(f" res.grad : {str(id(res.grad))[-6:]}, {res.grad is not None}")
print("=========================")
print()

s = ndl.autograd.find_topo_sort([res])

for n in s:
    print(f"{str(id(n))[-6:]}: {n.cached_data is None}")
print()

res.backward()

for n in s:
    print(f"{str(id(n))[-6:]}: {n.cached_data is None}")
print()

print("=========================")
print(f"        a : {str(id(a))[-6:]}")
print(f"        b : {str(id(b))[-6:]}")
print(f"        c : {str(id(c))[-6:]}")
print(f"        d : {str(id(d))[-6:]}")
print(f"      res : {str(id(res))[-6:]}")
print(f"   a.grad : {str(id(a.grad))[-6:]}, {a.grad is not None}")
print(f"   b.grad : {str(id(b.grad))[-6:]}, {b.grad is not None}")
print(f"   c.grad : {str(id(c.grad))[-6:]}, {c.grad is not None}")
print(f"   d.grad : {str(id(d.grad))[-6:]}, {d.grad is not None}")
print(f" res.grad : {str(id(res.grad))[-6:]}, {res.grad is not None}")
print("=========================")
print()

print(f"a.grad: {a.grad}")
res.backward()
print(f"a.grad: {a.grad}")
