
# coding: utf-8

# In[19]:


#Simple iteration method for roots of high degree equations
from math import sqrt
x=2 #the initial guess
for iteration in range(1,101): #100 iterations
    xnew = sqrt((5*x - 3)/2)
    print(iteration,x)
    if abs(xnew - x) < 0.000001:
        break
    x = xnew
print('The root : %0.5f' % xnew) #showing the number upto 5th decimal point
print('The number of iterations : %d' % iteration)


# In[23]:


#Simple iteration method for roots of high degree equations
from math import sqrt
x=3 #initial condition
for i in range(1,101): # for 100 iterations
    xnew = (2*x**2-3)/5
    print(i, x)
    if abs(xnew-x) < 0.000001:
        break
    x = xnew
print('The root : %0.5f' %xnew) #showing the number upto 5th decimal point
print('The no of iterations : %d' %i)


# In[24]:


#Simple iteration method for roots of high degree equations using while loop
x=5 #initial condition
xnew=0
i=0
while abs(xnew - x) >= 0.000001:
    i += 1
    x = xnew
    xnew = (2*x**2 + 3)/5
print('The root : %0.5f' %xnew)
print('The no of iterations : %d' %i)


# In[25]:


#Simple Newton Raphson method for roots of high degree equations
from math import *
x=0
for i in range(1,101):
    xnew = x - (2*x**2-5*x+3)/(4*x-5) # the Newton Raphson formula
    if abs(xnew - x) < 0.000001:
        break
    x = xnew
print('The root is: %0.5f' %xnew)
print('The no of iterations: %d' %i)


# In[33]:


#Simple bisection method for roots of high degree equations
x1 = 0
x2 = 2
y1 = (2 * x1**2 - 5 * x1 + 3)
y2 = (2 * x2**2 - 5 * x2 + 3)
if (y1*y2) > 0:
    print('The roots donot lie within the given interval')
    exit
for bisection in range(1,101): # 100 bisections
    xh = (x1 + x2)/2
    yh = (2 * xh**2 - 5 * xh + 3)
    y1 = (2 * x1**2 - 5 * x1 + 3)
    if abs(y1) < 1.0e-6:
        break
    elif (y1*yh) < 0: # if the root is in first half interval
        x2 = xh
    else:
        x1 = xh
print('The root is: %0.5f' %x1)
print('The no of bisections: %d' %bisection)
    


# In[46]:


#Regula Falsi method for roots of high degree equations
def Rfalsi(fn, x1, x2, tolerance=0.000001, i_limit=101):
    y1 = fn(x1)
    y2 = fn(x2)
    xh = 0
    i = 0 # for counting number of false positions
    if y1 == 0: xh = x1
    elif y2 == 0: xh = x2
    elif y1 * y2 > 0:
        print('No roots exist between the given interval')
    else:
        for i in range(1,i_limit):
            xh = x2 - (((x2 - x1) * y2)/(y2 - y1))
            yh = fn(xh)
            if abs(yh) < tolerance:
                break
            elif (y1 * yh) < 0:
                x2 = xh
                y2 = yh
            else:
                x1 = xh
                y1 = yh
    return xh, i

def y(x): return 2 * x**2 - 5 * x + 3
x1 = float(input('The value of x1: '))
x2 = float(input('The value of x2: '))
x,n = Rfalsi(y, x1, x2)
print('The root: %0.5f' %x)
print('The no of false positions: %d' %n)


# In[56]:


#Regula Falsi method for roots of high degree equations
from math import cos
def Rfalsi(fn, x1, x2, tolerance=0.000001, i_limit=101):
    y1 = fn(x1)
    y2 = fn(x2)
    xh = 0
    i = 1 # for counting number of false positions
    if y1 == 0: xh = x1
    elif y2 == 0: xh = x2
    elif y1 * y2 > 0:
        print('No roots exist between the given interval')
    else:
        for i in range(1,i_limit):
            xh = x2 - (x2 - x1)/(y2 - y1) * y2
            yh = fn(xh)
            if abs(yh) < tolerance:
                break
            elif (y1 * yh) < 0:
                x2 = xh
                y2 = yh
            else:
                x1 = xh
                y1 = yh
    return xh, i

y = lambda x: x**2 + cos(x)**2 - 4*x
x1 = float(input('The value of x1: '))
x2 = float(input('The value of x2: '))
x,n = Rfalsi(y, x1, x2)
print('The root: %0.5f' %x)
print('The no of false positions: %d' %n)


# In[61]:


#Secant method for roots of high degree equations
from math import cos
def secant(fn, x1, x2, tolerance=0.000001, maxiter=100):
    for iteration in range(maxiter):
        xnew = x2 - (x2-x1)/(fn(x2) - fn(x1)) * fn(x2)
        if abs(fn(xnew)) < tolerance: break
        else:
            x1 = x2
            x2 = xnew
    else: print('Warning: maximum number of iteration reached!!!')
    return xnew, iteration

f = lambda x: x**2 + cos(x)**2 - 4*x
x1 = float(input('The value of x1:'))
x2 = float(input('The value of x2:'))
r,n = secant(f,x1,x2)
print('The root is: %0.5f at iteration %d' %(r,n))
            


# In[73]:


#root finding functions in SciPy
from scipy.optimize import newton, bisect, fsolve, root
f = lambda x: 2*x**2 - 5 *x + 3
print(newton(f,0))
print(newton(f,2))
print(bisect(f, 1.1, 1.8))
print(fsolve(f, 0))
print(fsolve(f, 4))
x0 = [-1, 0, 1, 2, 3, 4]
print(fsolve(f, x0))
print(root(f,0).x)
print(root(f,x0).x)


# In[11]:


#linear interpolation and curve fitting
time = [0, 20, 40, 60, 80, 100]
temperature = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
temp70 = y(70, time, temperature)
print('The temperature at time 70 is: %0.5f' %temp70)

def y(xp, x, y):
    for i, xi in enumerate(x):
        if xp < xi:
            return y[i-1] + (y[i] - y[i-1])/(x[i] - x[i-1]) * (xp - x[i-1])
    else: 
        print('The given x is out of range')


# In[13]:


#lagrange interpolation and curve fitting
x = [0, 20, 40, 60, 80, 100]
y = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
m = len(x) #number of points or length of x list
n = m - 1
xp = float(input('The value of x :'))
yp = 0
for i in range(n+1):
    L = 1
    for j in range(n+1):
        if j != i:
            L *= (xp - x[j])/(x[i] - x[j])
    yp += y[i] * L
print('For x = %0.1f, y = %0.1f' %(xp, yp))


# In[10]:


#Newton's interpolation 
x = [0.0, 1.5, 2.8, 4.4, 6.1, 8.0]
y = [0.0, 0.9, 2.5, 6.6, 7.7, 8.0]
n = len(x) - 1
import numpy as np
Dy = np.zeros((n+1,n+1))
Dy[:,0] = y
for j in range(n):
    for i in range(j+1,n+1):
        Dy[i, j+1] =  (Dy[i, j] - Dy[j, j])/(x[i] - x[j])
print(Dy)

xp = float(input('The value of x:'))
yp = Dy[0,0]
for i in range(n):
    xprod = 1
    for j in range(i+1):
        xprod *= xp - x[j]
    yp += xprod * Dy[i+1, i+1]
print('For x = %0.1f, y = %0.1f' %(xp, yp))


# In[7]:


#Linear regression and curve fitting
x = [3, 4, 5, 6, 7, 8]
y = [0, 7, 17, 26, 35, 45]
n = len(x) #the number of given points or length of the array
sumx = sumy = sumxy = sumx2 = 0
for i in range(n):
    sumx += x[i]
    sumy += y[i]
    sumxy += x[i] * y[i]
    sumx2 += x[i]**2
xm = sumx / n
ym = sumy / n
a = (ym * sumx2 - xm * sumxy) / (sumx2 - n * xm**2)
b = (sumxy - xm * sumy) / (sumx2 - n * xm**2)
print('The straight line equation of the given points: ')
print('f(x) = (%0.3f) + (%0.3f) x' %(a,b))


# In[11]:


#Linear regression and curve fitting using numpy finctions
from numpy import array, sum, mean
x = array([3, 4, 5, 6, 7, 8], float)
y = array([0, 7, 17, 26, 35, 45], float)
n = len(x) #the number of given points or length of the array
a = (mean(y) * sum(x**2) - mean(x) * sum(x * y)) / (sum(x**2) - n * mean(x)**2)
b = (sum(x * y) - mean(x) * sum(y)) / (sum(x**2) - n * mean(x)**2)
print('The straight line equation of the given points: ')
print('f(x) = (%0.3f) + (%0.3f) x' %(a,b))


# In[18]:


#Polynomial curve fitting 
import numpy as np
x = np.arange(6)
y = np.array([2, 8, 14, 28, 39, 62], float)
m = len(x)
n = 3
A = np.zeros((n+1, n+1))
B = np.zeros(n+1)
a = np.zeros(n+1)
for row in range(n+1):
    for col in range(n+1):
        if row == 0 and col == 0:
            A[row, col] = m
            continue
        A[row, col] = np.sum(x**(row + col))
    B[row] = np.sum(x**row * y)
print(B)
a = np.linalg.solve(A,B)
print('The polynomial :')
print('f(x) = \t %0.3f' %a[0])
for i in range(1,n+1):
    print('\t %+0.3f x^%d' %(a[i], i))


# In[33]:


#interpolation functions of sciPy
from scipy.interpolate import interp1d, lagrange
x = [0, 20, 40, 60, 80, 100]
y = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
#f = interp1d(x, y)
f = interp1d(x, y, 'quadratic')
print(f(50))
L = lagrange(x, y)
print(L)
print(L(50))
print(L(40))


# In[47]:


#curve fitting functions of sciPy
from scipy.stats import linregress
x = [3, 4, 5, 6, 7, 8]
y = [0, 7, 17, 26, 35, 45]
L = linregress(x, y)
print(L)
print('The straight line equation of the given points:')
print('y = (%0.3f) + (%0.3f) * x' %(L.intercept, L.slope))

import numpy as np
x = np.arange(6)
y = np.array([2, 8, 14, 28, 39, 62], float)
from scipy.optimize import curve_fit
def f (x, a0, a1, a2): return a0 + a1 * x+ a2 * x**2
a, b = curve_fit(f, x, y)
print(a)

def f (x,a0, a1, a2, a3): 
    return a0 + a1 * x+ a2 * x**2 + a3 * x**3
a, b = curve_fit(f, x, y)
print(a)


# In[12]:


#Numerical differentiation Finite Difference Method
f = lambda x: 0.1 * x**5 - 0.2 * x**3 + 0.1 * x - 0.2
h = 0.05
x = 0.1
dff1 = (f(x + h) - f(x)) / h #forward FDM
dff2 = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / h**2
print('The first and second differentials by using FFDM:')
print('f\'(%f) = (%f)'%(x, dff1))
print('f\'\'(%f) = (%f)'%(x, dff2))

dfb1 = (f(x) - f(x - h)) / h #backward FDM
dfb2 = (f(x) - 2 * f(x - h) + f(x - 2 * h)) / h**2
print('The first and second differentials by using BFDM:')
print('f\'(%f) = (%f)'%(x, dfb1))
print('f\'\'(%f) = (%f)'%(x, dfb2))

dfc1 = (f(x + h) - f(x - h)) / (2 * h) #central FDM
dfc2 = (f(x + h) - 2 * f(x) + f(x - h)) / h**2
print('The first and second differentials by using CFDM:')
print('f\'(%f) = (%f)'%(x, dfc1))
print('f\'\'(%f) = (%f)'%(x, dfc2))


# In[9]:


#plotting derivative curves
import numpy as np
import matplotlib.pyplot as plt
f = lambda x: 0.1 * x**5 - 0.2 * x**3 + 0.1 * x - 0.2
h = 0.05
x = np.linspace(0, 1, 11)
dff1 = (f(x + h) - f(x)) / h #forward FDM
dff2 = (f(x + 2 * h) - 2 * f(x + h) + f(x)) / h**2
print('The first and second differentials by using FFDM:')
print(dff1)
print(dff2)

dfb1 = (f(x) - f(x - h)) / h #backward FDM
dfb2 = (f(x) - 2 * f(x - h) + f(x - 2 * h)) / h**2
print('The first and second differentials by using BFDM:')
print(dfb1)
print(dfb2)

dfc1 = (f(x + h) - f(x - h)) / (2 * h) #central FDM
dfc2 = (f(x + h) - 2 * f(x) + f(x - h)) / h**2
print('The first and second differentials by using CFDM:')
print(dfc1)
print(dfc2)

plt.plot(x, f(x),'-k', x, dff1, '--b', x, dff2, '-.r', x, dfb1, '-*y', x, dfb2, '-*g', x, dfc1, '-or', x, dfc2, '-ob')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['f(x)', 'dff1', 'dff2', 'dfb1', 'dfb2' ,'dfc1', 'dfc2'])
plt.grid()
plt.show()


# In[19]:


#differentiation using SciPy function
import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt
f = lambda x: 0.1 * x**5 - 0.2 * x**3 + 0.1 * x - 0.2
x = np.linspace(0, 1, 11)
h = 0.05
y2 = derivative(f, x, h, 2)
print(y2) 
plt.plot(x, f(x), '-k', x, y2, '-or')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['f(x)', 'y2'])
plt.grid()
plt.show()


# In[29]:


#Trapezoidal rule for numerical integration
from math import sin, pi
f = lambda x: x * sin(x)
a = 0
b = pi / 2
n = 100
h = (b - a) / n
S = 0.5 * (f(a) + f(b))
for i in range(1, n):
    S += f(a + i * h)
integral = h * S
print('Integral = %f' %integral)


# In[37]:


#Simpson's 1/3rd rule for numerical integration
from math import sin, pi
f = lambda x: x * sin(x)
a = 0
b = pi / 2
n = 18
h = (b - a) / n
S = f(a) + f(b)
for i in range(1, n, 2):
    S += 4 * f(a + i * h)
for i in range(2, n, 2):
    S += 2 * f(a + i * h)
Integral = h / 3 * S
print('The integral = %0.8f' %Integral)


# In[42]:


#Simpson's 3/8th rule for numerical integration
from math import sin, pi
f = lambda x: x* sin(x)
a = 0
b = pi / 2
n = 18
h = (b-a) / n
S = f(a) + f(b)
for i in range(1, n, 3):
    S += 3 * (f(a + i * h) + f(a+ (i+1) * h))
for i in range(3, n, 3):
    S += 2 * f(a + i * h)
Integral = 3 / 8 * h * S
print('The integral = %0.8f' %Integral)


# In[93]:


#double integration for numerical integration
f = lambda x, y: x**2 * y + x * y**2
ax = 1
bx = 2
ay = -1
by = 1
nx = 10
ny = 10
hx = (bx - ax) / nx
hy = (by - ay) / ny
S = 0
for i in range(ny+1):
    if i == 0 or i == ny:
        p = 1
    elif i % 2 == 1:
        p = 4
    else: 
        p = 2
    for j in range(nx+1):
        if j == 0 or j == nx:
            q = 1
        elif j % 2 == 1:
            q = 4
        else:
            q = 2
        S += p * q * f(ax + j * hx, ay + i * hy) 
Integral = hx * hy / 9 * S
print('Integral = %f' %Integral)


# In[110]:


#Quadrature in SciPy for numerical integration
import numpy as np
from scipy.integrate import quad, dblquad, nquad
f = lambda x: x * np.sin(x)
print(quad(f, 0, np.pi/2))
I,_ = quad(f, 0, np.pi/2)
print(I)
f = lambda x, y: x**2 * y + x * y**2
ax = 1
bx = 2
ay = -1
by = 1
I = dblquad(f, ax, bx, lambda y: ay, lambda y: by)
print(I)
I,_ = dblquad(f, ax, bx, lambda y: ay, lambda y: by)
print(I)
I = nquad(f, [[ax, bx], [ay, by]])
print(I)
I,_ = nquad(f, [[ax, bx], [ay, by]])
print(I)


# In[2]:


#gauss elimination & back substitution for solving system of linear equations
from numpy import array, zeros
A = array([[0, 7, -1, 3, 1], 
           [2, 3, 4, 1, 7], 
           [6, 2, 0, 2, -1], 
           [2, 1, 2, 0, 2], 
           [3, 4, 1, -2, 1]], float)
B = array([5, 7, 2, 3, 4], float)
n = len(B)
x = zeros(n, float)
for i in range(n-1): #elimination
    if A[i, i] == 0:
        for j in range(n):
            A[i, j], A[i+1, j] = A[i+1, j], A[i, j]
        B[i], B[i+1] = B[i+1], B[i]
    for j in range(i+1, n):
        if A[j, i] == 0:
            continue
        factor = A[i, i] / A[j, i]
        B[j] = B[i] - factor * B[j]
        for k in range(i, n):
            A[j, k] = A[i, k] - factor * A[j, k]
print('The matrix A after transformation to upper triangular matrix= ')
print(A)
print('The matrix B =')
print(B)

x[n-1] = B[n-1] / A[n-1, n-1] #back substitution
for i in range(n-2, -1, -1):
    terms = 0
    for j in range(i+1, n):
        terms += A[i, j] * x[j]
    x[i] = (B[i] - terms) / A[i, i]
print('The solution of the system of linear equation is =')
print(x)


# In[8]:


#jacobi method for solving system of linear equations
import numpy as np
a = np.array([[4, 1, 2, -1], 
             [3, 6, -1, 2], 
             [2, -1, 5, -3], 
             [4, 1, -3, -8]], float)
b = np.array([2, -1, 3, 2], float)
(n,) = np.shape(b)
x = np.full(n, 1.0, float) #initial value of x is 1.0
xnew = np.empty(n, float)
itermax = 100
tolerance = 1.0e-6
#iterations:
for iterations in range(itermax):
    for i in range(n):
        sum = 0
        for j in range(n):
            if j != i:
                sum += a[i, j] * x[j]
        xnew[i] = -1 / a[i, i] * (sum - b[i])
    if (abs(xnew - x) < tolerance).all():
        break
    else:
        x = np.copy(xnew)
print('The solution is:')
print(x)
print('Number of iterations: %d' %(iterations+1))

