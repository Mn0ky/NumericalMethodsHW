import sympy as sp

# Define the variable
t = sp.symbols('t', real=True)

# Define the components of the vector
x = t
y = (t**2)
z = (sp.Rational(2, 3)*t**3)

# Create the vector
r_vector = sp.Matrix([x, y, z])
r_vector_dt = r_vector.diff(t)

t_vector = r_vector_dt/r_vector_dt.norm()
t_vector_dt = t_vector.diff()

normal_vector = t_vector_dt/t_vector_dt.norm()
binormal_vector = t_vector.cross(normal_vector)

print(sp.simplify(t_vector.subs(t, 4)))
print(sp.simplify(normal_vector.subs(t, 4)))
print(sp.simplify(binormal_vector.subs(t, 4)))