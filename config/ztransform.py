from sympy import symbols, summation, KroneckerDelta, oo
import streamlit as st

def manual_z_transform(f, n, z):
    return summation(f * z**(-n), (n, 0, oo))

def z_transform_examples():
    n, z = symbols('n z')
    f = 1**n
    f1 = 2**n
    f2 = KroneckerDelta(n, 0)
    f3 = 5 * KroneckerDelta(n, 0)
    
    result1 = manual_z_transform(f, n, z)
    result2 = manual_z_transform(f1, n, z)
    result3 = manual_z_transform(f2, n, z)
    result4 = manual_z_transform(f3, n, z)
    
    st.write("Z-Transform of 1^n:", result1)
    st.write("Z-Transform of 2^n:", result2)
    st.write("Z-Transform of KroneckerDelta:", result3)
    st.write("Z-Transform of 5 * KroneckerDelta:", result4)
