#In a .py script (not a notebook — this is reusable code with tests), implement these from scratch using only basic loops/arithmetic (no NumPy inside your functions):
# 1 - dot_product(a, b)
import math
import numpy as np
def dot_product(a: list, b: list):
    output = 0
    if len(a) != len(b):
        print("size should be equal")
    else:
        output=sum(a*b for a, b in zip(a,b))                   # this is for matrices        for element in a:                                                                                   
    return output                                     # for elements in b:
a = [1,2]                                              #multiply = elements * element
b=[1,3]                                                # output+=multiply
print(f"dot product of {a} and {b} is=", dot_product(a,b))                                       

# now for the magnitude of a vector: (imported math for this)
# 2- magnitude(v) — the length of a vector, sqrt(sum of squares) 
def magnitude(c: list):
    result = 0
    process=sum(c*c for c in c)
    result= math.sqrt(process)                                                                                                    
    return result                          
c= [4,5]
print(f"the magnitude of {c} is=", magnitude(c))

# 3- cosine_similarity(a, b) — dot_product(a,b) / (magnitude(a) * magnitude(b))
def cosine_similarity(u: list, v: list):
    final = 0
    magnitudes = magnitude(u) * magnitude(v)
    final = dot_product(u,v) / magnitudes
    return final
u = [1, 6]
v = [5, 2]
print(f"cosine similarity of {u} and {v}=", cosine_similarity(u,v))

# 4- mean(v)
def vec_mean(p: list):
    last = 0
    for num in p:
        last += num
    last1 = last / len(p)
    return last1
p = [4, 2]
print(f"the mean of {p}=", vec_mean(p))

# now for variance
def vec_variance(t: list):
    step3 = 0
    step4 = 0
    for numb in t:
        step1 = vec_mean(t)
        step2 = numb - step1
        step3 += step2**2
        step4 = step3 / len(t)
    return step4
t = [ 2 , 8]
print(f" variance of {t} is", vec_variance(t))

# q20: Then verify each against NumPy (np.dot, np.linalg.norm, etc.) on random inputs
# import numpy now at the st
print("--------------------------------------------\n", "now we'll verify each with numpy")
print(f"first for dot product of {a} and {b} with numpy:", np.dot(a,b))
print(f"second is the magnitude of {c} with numpy:", np.linalg.norm(c))
print(f"third is cosine similarity of {v} and {v} with numpy:", np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
print(f"fourth is mean of {p} with numpy:", np.mean(p))
print(f"lastly is variance of {t} with numpy:", np.var(t))

# q21: Write simple assertions that compare your result to NumPy's within a tiny tolerance (e.g. abs(mine - numpy) < 1e-9).
print("--------------------------------------------\n", "low tolerance test")
v1 = [ 6 , 8 ]
v2 = [ 9 , 4 ]
vec1 = np.array(v1)
vec2 = np.array(v2)
tolerance = 1e-9
mine_dot= (dot_product(v1,v2))
numpy_dot= np.dot(vec1, vec2)
print(f"dot product of {v1} and {v2}: success if silent")
assert abs(mine_dot - numpy_dot) < tolerance, "dot product failed"

mine_magnitude= (magnitude(v1))
numpy_magnitude= np.linalg.norm(vec1)
print(f"magnitude of {v1}: success if silent")
assert abs(mine_magnitude - numpy_magnitude) < tolerance, "magnitude failed"

mine_cos= (cosine_similarity(v1,v2))
numpy_cos= np.dot(v1,v2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
print(f"cos similarity of {v1} and {v2}: success if silent")
assert abs(mine_cos - numpy_cos) < tolerance, "cosine similarity failed"

mine_mean= (vec_mean(v1))
numpy_mean= np.mean(vec1)
print(f"mean of {v1}: success if silent")
assert abs(mine_mean - numpy_mean) < tolerance, "mean failed"

mine_var= (vec_variance(v1))
numpy_var= np.var(vec1)
print(f"variance of {v1}: success if silent")
assert abs(mine_var - numpy_var) < tolerance, "variance failed"