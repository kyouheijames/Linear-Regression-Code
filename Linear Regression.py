#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Weighted Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import least_squares
import scipy as sp

a = 0
b = 0
delta = 0

x = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70])
y = np.array([-33,-31,-29,-27,-25,-23.5,-22,-20,-18,-16.5,-14.5,-13,-11,-9,-7])
sigma = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
w = 1/sigma**2

wxy = sum(w*x*y)
wy = sum(w*y)
wx = sum(w*x)
wx2 = sum(w*(x**2))
wy2 = sum(w*y**2)
ws = sum(w)
delta = ws*wx2 - wx**2
wm = wx/ws
wmy = wy/ws
errorb= np.sqrt(ws/delta)
errora= np.sqrt(wx2/delta)

a = (wx2*wy - wx*wxy)/delta
b = (ws*wxy - wx*wy)/delta

ds = y - (a+b*x)

Chi2 = sum((y - (a +b*x))**2*w**2)

plt.errorbar(x, y, sigma, fmt='o')
plt.plot(x , a + b*x)
plt.show()

wstd = np.sqrt(5/4*(wy2 - ws*wmy**2)/ws)


er1 = sp.special.erf((1-a)/np.sqrt(2*errora**2))
er2 = sp.special.erf((-1-a)/np.sqrt(2*errora**2))

prob =(er1 - er2)/2

cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
xm = sum(x)/x.size
ym = sum(y)/x.size
xym = sum(x*y)/x.size
x2m = sum(x**2)/x.size
var = sum(x**2)/x.size - (sum(x)/x.size)**2
rchi2 = Chi2/(15-2)
r2 = 1 - (sum((y-(a+b*x))**2)/(sum((y - ym)**2)))
ar2 = r2 - 1/(x.size-2)*(1-r2)


print('Intercept is:',a, '+/-', errora, 'Slope is',b, '+/-' , errorb,'Weighted Mean is:', wm)
print('Height at the weighted mean is:', a + b*wm, 'Chi-Squared is:',Chi2)
print('Degrees of Freedom = Number of Data points - Parameters:', x.size - 2)
print('Probability is', prob, 'Covariance is', cov, 'x mean is', xm)
print('y mean is', ym, 'x squared mean is',x2m, 'variance is', var, "xy mean is", xym)
print('Adjusted r^2 value is', ar2, 'Reduced chi-squared is', rchi2,'r2',r2)
print(a + b*80, '+/-', errora + 80*errorb)


# In[8]:


#Exp 2 Glass

import numpy as np
import matplotlib.pyplot as plt

datax = np.array([0, 0.6, 0.85, 1.33, 1.7, 2.15, 2.5, 2.88])
datax1 = np.array([0.6, 0.85, 1.33, 1.7, 2.15, 2.5, 2.88])
y = datax/np.sqrt(datax**2+4*1.8**2)

a = 0
b = 0
delta = 0

x = np.array([np.sin(0),np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
sigma = 4*1.8**2/((np.sqrt(4*1.8**2+x**2))**3)*0.05
w = 1/sigma**2

wxy = sum(w*x*y)
wy = sum(w*y)
wx = sum(w*x)
wx2 = sum(w*(x**2))
wy2 = sum(w*y**2)
ws = sum(w)
delta = ws*wx2 - wx**2
wm = wx/ws
wmy = wy/ws
errorb= np.sqrt(ws/delta)
errora= np.sqrt(wx2/delta)

a = (wx2*wy - wx*wxy)/delta
b = (ws*wxy - wx*wy)/delta

ds = y - (a+b*x)

Chi2 = sum((y - (a +b*x))**2*w)

plt.errorbar(x, y, sigma, fmt='o')
plt.plot(x , a + b*x)
plt.show()

wstd = np.sqrt(5/4*(wy2 - ws*wmy**2)/ws)

cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
xm = sum(x)/x.size
ym = sum(y)/x.size
xym = sum(x*y)/x.size
x2m = sum(x**2)/x.size
var = sum(x**2)/x.size - (sum(x)/x.size)**2
rchi2 = Chi2/(x.size-2)
r2 = 1 - (sum((y-(a+b*x))**2)/(sum((y - ym)**2)))
ar2 = r2 - 1/(x.size-2)*(1-r2)

x1 = np.array([np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
y1 = datax1/np.sqrt(datax1**2+4*1.8**2)

nb = x1/y1
print('Intercept is:',a, '+/-', errora, 'Slope is',b, '+/-' , errorb,'Weighted Mean is:', wm , ' Height at the weighted mean is:', a + b*wm, 'Chi-Squared is:', Chi2, 'Degrees of Freedom = Number of Data points - Parameters:', x.size - 2)
print('Covariance is', cov, 'x mean is', xm, 'y mean is', ym, 'x squared mean is',x2m, 'variance is', var, "xy mean is", xym, 'Adjusted r^2 value is', ar2, 'Reduced chi-squared is', rchi2,'r2',r2, a + b*80, '+/-', errora + 80*errorb)
print(nb)


# In[31]:


#Exp 2 Acryllic Weighted

import numpy as np
import matplotlib.pyplot as plt

datax = np.array([0, 0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
datax1 = np.array([0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
y = datax/np.sqrt(datax**2+4*1.8**2)

a = 0
b = 0
delta = 0

x = np.array([np.sin(0),np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
dx = 1/360*np.pi*np.array([np.cos(0),np.cos(np.pi/180*10),np.cos(np.pi/180*20),np.cos(np.pi/180*30),np.cos(np.pi/180*40),np.cos(np.pi/180*50),np.cos(np.pi/180*60),np.cos(np.pi/180*70)])
sigma = 4*1.8**2/((np.sqrt(4*1.8**2+x**2))**3)*0.05
w = 1/sigma**2

wxy = sum(w*x*y)
wy = sum(w*y)
wx = sum(w*x)
wx2 = sum(w*(x**2))
wy2 = sum(w*y**2)
ws = sum(w)
delta = ws*wx2 - wx**2
wm = wx/ws
wmy = wy/ws

b = wxy/wx2
errorb= np.sqrt(1/(x.size-1)*sum((y-b*x)**2/sum(x**2)))

Chi2 = sum((y - (b*x))**2*w)

x1 = np.array([np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
y1 = datax1/np.sqrt(datax1**2+4*1.8**2)
nb = x1/y1

cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
xm = sum(x)/x.size
ym = sum(y)/x.size
xym = sum(x*y)/x.size
x2m = sum(x**2)/x.size
var = sum(x**2)/x.size - (sum(x)/x.size)**2
rchi2 = Chi2/(x.size-2)
r2 = 1 - (sum((y-(a+b*x))**2)/(sum((y - ym)**2)))
ar2 = r2 - 1/(x.size-2)*(1-r2)
varx = sum(x**2)/x.size - (sum(x)/x.size)**2
vary = sum(y**2)/x.size - (sum(y)/x.size)**2
r = cov/np.sqrt(varx*vary)
delz = np.sqrt(sigma**2+b*dx**2)

plt.errorbar(x, y, delz, fmt='o')
plt.plot(x , b*x)
plt.plot(x , (b+errorb)*x, linestyle = 'dashed')
plt.plot(x , (b-errorb)*x, linestyle = 'dashed')
plt.show()


print('Slope is',b, '+/-' , errorb,'Weighted Mean is:', wm , ' Height at the weighted mean is:', a + b*wm, 'Chi-Squared is:', Chi2, 'Degrees of Freedom = Number of Data points - Parameters:', x.size - 2)
print('Covariance is', cov, 'x mean is', xm, 'y mean is', ym, 'x squared mean is',x2m, 'variance is', var, "xy mean is", xym, 'Adjusted r^2 value is', ar2, 'Reduced chi-squared is', rchi2,'r2',r2)
print(nb, r, delz)


# In[21]:


#Exp Acryllic Simple

import numpy as np
import matplotlib.pyplot as plt

datax = np.array([0, 0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
datax1 = np.array([0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
y = datax/np.sqrt(datax**2+4*1.8**2)
sigma = 4*1.8**2/((np.sqrt(4*1.8**2+x**2))**3)*0.05

m = 0
delta = 0

x = np.array([np.sin(0),np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
dx = 1/360*np.pi*np.array([np.cos(0),np.cos(np.pi/180*10),np.cos(np.pi/180*20),np.cos(np.pi/180*30),np.cos(np.pi/180*40),np.cos(np.pi/180*50),np.cos(np.pi/180*60),np.cos(np.pi/180*70)])
xy = sum(x*y)
ys = sum(y)
xs = sum(x)
x2 = sum(x**2)
y2 = sum(y**2)

m = xy/x2

ds = y - (m*x)

dy = np.sqrt(sum(ds**2)/(x.size-1))
dm = dy/np.sqrt(x.size*x2)



x1 = np.array([np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
y1 = datax1/np.sqrt(datax1**2+4*1.8**2)
nb = x1/y1

cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
xm = sum(x)/x.size
ym = sum(y)/x.size
xym = sum(x*y)/x.size
x2m = sum(x**2)/x.size
var = sum(x**2)/x.size - (sum(x)/x.size)**2
r2 = 1 - (sum((y-(m*x))**2)/(sum((y - ym)**2)))
ar2 = r2 - 1/(x.size-2)*(1-r2)
varx = sum(x**2)/x.size - (sum(x)/x.size)**2
vary = sum(y**2)/x.size - (sum(y)/x.size)**2
r = cov/np.sqrt(varx*vary)
delz = np.sqrt(sigma**2+m*dx**2)
plt.errorbar(x, y, delz, fmt='o')
plt.plot(x , m*x)
plt.show()

print('Slope is',m, '+/-' , dm,'Weighted Mean is:', wm , ' Height at the weighted mean is:',m*xm, 'Degrees of Freedom = Number of Data points - Parameters:', x.size - 2)
print('Covariance is', cov, 'x mean is', xm, 'y mean is', ym, 'x squared mean is',x2m, 'variance is', var, "xy mean is", xym, 'Adjusted r^2 value is', ar2)
print(nb, r, delz)


# In[32]:


#Exp 4
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0,10,20,30,40,50,60,70,80,90])
y = np.array([0,10.5,20,29.5,40,50,60,70,79.5,90])
sigma = np.array([0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75])
w = 1/sigma**2
Chi2 = sum((y - x)**2*w)
rchi2 = Chi2/(x.size)
varx = sum(x**2)/x.size - (sum(x)/x.size)**2
vary = sum(y**2)/x.size - (sum(y)/x.size)**2
cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
r = cov/np.sqrt(varx*vary)

plt.errorbar(x, y, sigma, fmt='o')
plt.plot(x , y)
plt.show()
print(rchi2,Chi2,r,cov)


# In[36]:


#Exp 5
import numpy as np
import matplotlib.pyplot as plt

datax = np.array([0, 0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
datax1 = np.array([0.52, 0.84, 1.30, 1.82, 2.22, 2.64, 3])
x = np.array([np.sin(0),np.sin(np.pi/180*7),np.sin(np.pi/180*13.5),np.sin(np.pi/180*20),np.sin(np.pi/180*26),np.sin(np.pi/180*31),np.sin(np.pi/180*36),np.sin(np.pi/180*39),np.sin(np.pi/180*42)])
sigma = 1/360*np.pi*np.array([np.cos(0),np.cos(np.pi/180*7),np.cos(np.pi/180*13.5),np.cos(np.pi/180*20),np.cos(np.pi/180*26),np.cos(np.pi/180*31),np.cos(np.pi/180*36),np.cos(np.pi/180*39),np.cos(np.pi/180*42)])
dx = 1/360*np.pi*np.array([np.cos(0),np.cos(np.pi/180*10),np.cos(np.pi/180*20),np.cos(np.pi/180*30),np.cos(np.pi/180*40),np.cos(np.pi/180*50),np.cos(np.pi/180*60),np.cos(np.pi/180*70),np.cos(np.pi/180*80)])

m = 0
delta = 0

y = np.array([np.sin(0),np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70),np.sin(np.pi/180*80)])

xy = sum(x*y)
ys = sum(y)
xs = sum(x)
x2 = sum(x**2)
y2 = sum(y**2)

m = xy/x2

ds = y - (m*x)

dy = np.sqrt(sum(ds**2)/(x.size-1))
dm = dy/np.sqrt(x.size*x2)



x1 = np.array([np.sin(np.pi/180*10),np.sin(np.pi/180*20),np.sin(np.pi/180*30),np.sin(np.pi/180*40),np.sin(np.pi/180*50),np.sin(np.pi/180*60),np.sin(np.pi/180*70)])
y1 = datax1/np.sqrt(datax1**2+4*1.8**2)
nb = x1/y1

cov = (sum(x*y)/x.size - sum(x)/x.size*sum(y)/x.size)
xm = sum(x)/x.size
ym = sum(y)/x.size
xym = sum(x*y)/x.size
x2m = sum(x**2)/x.size
var = sum(x**2)/x.size - (sum(x)/x.size)**2
r2 = 1 - (sum((y-(m*x))**2)/(sum((y - ym)**2)))
ar2 = r2 - 1/(x.size-2)*(1-r2)
varx = sum(x**2)/x.size - (sum(x)/x.size)**2
vary = sum(y**2)/x.size - (sum(y)/x.size)**2
r = cov/np.sqrt(varx*vary)
delz = np.sqrt(sigma**2+m*dx**2)
plt.errorbar(x, y, delz, fmt='o')
plt.plot(x , m*x)
plt.show()

print('Slope is',m, '+/-' , dm,'Weighted Mean is:', wm , ' Height at the weighted mean is:',m*xm, 'Degrees of Freedom = Number of Data points - Parameters:', x.size - 2)
print('Covariance is', cov, 'x mean is', xm, 'y mean is', ym, 'x squared mean is',x2m, 'variance is', var, "xy mean is", xym, 'Adjusted r^2 value is', ar2)
print(nb, r, delz)


# In[ ]:




