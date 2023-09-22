
import osqp
import numpy as np
from scipy import sparse


delta=0.1
a1=6
a2=8
alphax=0
alphay=0
alphaz=0
xddr=1
yddr=1
zddr=1
xdr=1
ydr=1
zdr=1
xr=0
yr=0
zr=0
xdd=1.1
ydd=1
zdd=1
xd=1
yd=1
zd=1
x=0
y=0.1
z=0

def solveOSQP(xddr,yddr,zddr,xdr,ydr,zdr,xr,yr,zr,xdd,ydd,zdd,xd,yd,zd,x,y,z,alphax,alphay,alphaz,a1,a2,delta):
    m=osqp.OSQP()
    P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
    #alphax etc are obtained with Klqrx*(xr-x)
    q=2*np.array([-alphax-xddr, -alphay-yddr, -alphaz-zddr])

    l=np.array([xddr+a1*(xdr-xd)+a2*(xr-x-delta),
       yddr+a1*(ydr-yd)+a2*(yr-y-delta),
       zddr+a1*(zdr-zd)+a2*(zr-z-delta)])

    u=np.array([xddr+a1*(xdr-xd)+a2*(xr-x+delta),
       yddr+a1*(ydr-yd)+a2*(yr-y+delta),
       zddr+a1*(zdr-zd)+a2*(zr-z+delta)])
    A = sparse.csc_matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
    m.setup(P=P, q=q, A=A, l=l, u=u, alpha=0.1)
    results=m.solve()
    return results
#m.update(q=q_new, l=l_new, u=u_new)

res=solveOSQP(xddr,yddr,zddr,xdr,ydr,zdr,xr,yr,zr,xdd,ydd,zdd,xd,yd,zd,x,y,z,alphax,alphay,alphaz,a1,a2,delta)
sol=res.x
print(sol)
print(type(sol))
print(np.size(sol))

# import osqp
# import numpy as np
# from scipy import sparse

# # Define problem data
# P = sparse.csc_matrix([[4, 1], [1, 2]])
# q = np.array([1, 1])
# A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
# l = np.array([1, 0, 0])
# u = np.array([1, 0.7, 0.7])

# # Create an OSQP object
# prob = osqp.OSQP()

# # Setup workspace and change alpha parameter
# prob.setup(P, q, A, l, u, alpha=1.0)

# # Solve problem
# res = prob.solve()