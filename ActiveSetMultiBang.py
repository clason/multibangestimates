"""
This module implements the active set method used in the paper
 'Error estimates for the approximation of a discrete-valued optimal 
  control problem'
by
 Christian Clason <christian.clason@uni-due.de>
 Thi Bich Tram Do <tram.do@uni-due.de>
 Frank Poerner <frank.poerner@mathematik.uni-wuerzburg.de>
see http://arxiv.org/abs/1803.04298
"""

__author__ = "Frank Poerner <frank.poerner@mathematik.uni-wuerzburg.de>"
__date__ = "March 12, 2018"

from fenics import *
from dolfin import *
import sys,os
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import *
from scipy.sparse.linalg import cgs
from scipy.sparse.linalg import spsolve
import numpy as np
import ufl
import mshr
import copy

# Example
# 1 = One-dimensional example 
# 2 = Two-dimensional example
example = 1

# Set linear algebra backend
parameters['linear_algebra_backend']='Eigen'

# Help function for assembling matrizes using the scipy structure
def assemble_scipy(a,bcs):
    aa = assemble(a)
    bcs.apply(aa)
    row,col,val = as_backend_type(aa).data()
    return sp.csr_matrix((val,col,row))

#set_log_level(30)
set_log_active(False)

# Number of outer iterations
maxiter = 10

# Number of maximal iterations for the active set method
max_inner_iter = 5


# Regularization parameter for the multi-bang regularization
alpha = 2.0

# The exact solution for the control is a discontinuous function, hence the errornorm function is not suited for computing
# the L^2 error. We compute the solution on a finer grid using a quadrature rule
Nfine = 1500
if (example==1):
    mesh_fine = UnitIntervalMesh(10000000)
if (example==2):
    domain_fine = mshr.Circle( Point(0.,0.), 1.0 )
    mesh_fine = mshr.generate_mesh(domain_fine, Nfine, "cgal")   
V_fine    = FunctionSpace(mesh_fine,'CG',1)

# Compute the solution on different meshes
for N in [75,250,750]:

# Construction of Function space
    if (example==1):
        mesh = UnitIntervalMesh(N)
    if (example==2):
        domain = mshr.Circle( Point(0.,0.), 1.0 )
        mesh = mshr.generate_mesh(domain, N, "cgal")
        
    V  = FunctionSpace(mesh,'CG',1)
    n  = V.dim()
    bc = DirichletBC(V, 0.0, "on_boundary")

    print("N: %d" % N)
    print("N_h (grid): %d" % n)
    print("N_h (fine grid): %d" % V_fine.dim())

    print "starting computation:"

    # Arrays to store the computed errors
    err_gamma = []
    err_reg = []

    # Control, adjoint, subgradient
    u,y,p,zeta = Function(V),Function(V),Function(V),Function(V)

    # Active sets (here d=5)
    Q1, Q2, Q3, Q4, Q5 = Function(V),Function(V),Function(V),Function(V),Function(V)

    # "Inactive" sets
    Q1_2, Q2_3, Q3_4, Q4_5 =Function(V),Function(V),Function(V),Function(V)

    temp = Function(V)
    rhs_zeta = Function(V)

    # build identity matrix
    I = sp.identity(n,format='csr')   
    zero_vector = Function(V)
    zero_vector.vector()[:]=0.0

    # define bilinear form
    w = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(w),grad(v))*dx

    # Construction of test example
    u1 = -2.0
    u2 = -1.0
    u3 = 0.0
    u4 = 1.0
    u5 = 2.0

    # One dimensional example
    if (example==1):
        # initialize optimal control
        expression_uopt ='''
        class my_expression : public Expression
        {
        public:

        void eval(Array<double>& values, const Array<double>& x) const
        {
            if (x[0] < 2.0/27.0)
            {
            values[0] = 0.0;
            }
            else if (x[0] < 2.0/9.0)
            {
            values[0] = 1.0;
            }
            else if (x[0] < 3.0/9.0)
            {
            values[0] = 2.0;
            }
            else if (x[0] < 4.0/9.0)
            {
            values[0] = 1.0;
            }
            else if (x[0] < 5.0/9.0)
            {
            values[0] = 0.0;
            }
            else if (x[0] < 6.0/9.0)
            {
            values[0] = -1.0;
            }
            else if (x[0] < 7.0/9.0)
            {
            values[0] = -2.0;
            }
            else if (x[0] < 25.0/27.0)
            {
            values[0] = -1.0;
            }
            else if (x[0] <= 1.0)
            {
            values[0] = 0.0;
            }
            else
            {
            values[0] = 0;
            }
        }
        };
        '''
        uopt_expr = Expression(expression_uopt, degree=1)
        uopt = interpolate(uopt_expr,V)

        # initialize optimal adjoint state
        expression_popt ='''
        class my_expression : public Expression
        {
        public:
        void eval(Array<double>& values, const Array<double>& x) const
        {
            if (x[0] < 2.0/9.0)
            {
            values[0] = 27.0/2.0*x[0];
            }
            else if (x[0] < 3.0/9.0)
            {
            values[0] = -72.0 + 3123.0/2.0*x[0] - 13122.0*pow(x[0],2) + 54675.0*pow(x[0],3) - 111537.0*pow(x[0],4) + 177147.0/2.0*pow(x[0],5);
            }
            else if (x[0] < 6.0/9.0)
            {
            values[0] = 9.0 - 18.0*x[0];
            }
            else if (x[0] < 7.0/9.0)
            {
            values[0] = -20079.0 + 136062.0*x[0] - 367416.0*pow(x[0],2) + 494262.0*pow(x[0],3) - 662661.0/2.0*pow(x[0],4) + 177147.0/2.0*pow(x[0],5);
            }
            else
            {
            values[0] = -27.0/2.0 + 27.0/2.0*x[0];
            }
        }
        };
        '''
        popt_expr = Expression(expression_popt, degree=1)
        popt = interpolate(popt_expr,V)

        # initialize laplace of optimal adjoint state
        expression_laplacepopt ='''
        class my_expression : public Expression
        {
        public:
        void eval(Array<double>& values, const Array<double>& x) const
        {
            if (x[0] < 2.0/9.0)
            {
            values[0] = 0.0;
            }
            else if (x[0] < 3.0/9.0)
            {
            values[0] = -26244.0 + 328050.0*x[0] - 1338444*pow(x[0],2) + 1771470.0*pow(x[0],3);
            }
            else if (x[0] < 6.0/9.0)
            {
            values[0] = 0.0;
            }
            else if (x[0] < 7.0/9.0)
            {
            values[0] = -734832.0 + 2965572.0*x[0]-3975966.0*pow(x[0],2) + 1771470.0*pow(x[0],3);
            }
            else
            {
            values[0] = 0.0;
            }
        }
        };
        '''
        laplacepopt_expr = Expression(expression_laplacepopt, degree=1)
        laplacepopt = interpolate(laplacepopt_expr,V)

        # initialize optimal state and its laplacian
        yopt = interpolate(Expression('sin(2*pi*x[0])', degree=1),V)
        laplaceyopt = interpolate(Expression('-4*pi*pi*sin(2*pi*x[0])', degree=1),V)


    # Two-dimensional example
    if (example==2):
        
        # initialize optimal control
        expression_uopt ='''
        class my_expression : public Expression
        {
        public:
        void eval(Array<double>& values, const Array<double>& x) const
        {
            double r = sqrt( pow(x[0],2) + pow(x[1],2) );
            if (r < 2.0/27.0)
            {
            values[0] = 0.0;
            }
            else if (r < 2.0/9.0)
            {
            values[0] = 1.0;
            }
            else if (r < 3.0/9.0)
            {
            values[0] = 2.0;
            }
            else if (r < 4.0/9.0)
            {
            values[0] = 1.0;
            }
            else if (r < 5.0/9.0)
            {
            values[0] = 0.0;
            }
            else if (r < 6.0/9.0)
            {
            values[0] = -1.0;
            }
            else if (r < 7.0/9.0)
            {
            values[0] = -2.0;
            }
            else if (r < 25.0/27.0)
            {
            values[0] = -1.0;
            }
            else if (r <= 1.0)
            {
            values[0] = 0.0;
            }
            else
            {
            values[0] = 0;
            }
        }
        };
        '''
        uopt_expr = Expression(expression_uopt, degree=1)
        uopt = interpolate(uopt_expr,V)

        # initialize optimal adjoint state
        expression_popt ='''
        class my_expression : public Expression
        {
        public:
        void eval(Array<double>& values, const Array<double>& x) const
        {
           double r = sqrt( pow(x[0],2) + pow(x[1],2) );
           if (r < 2.0/27.0)
           {
           values[0] = (59049.0*r*r*r)/4.0 - (531441.0*r*r*r*r)/2.0 + (43046721*r*r*r*r*r)/32.0;
           }
           else if (r < 2.0/9.0)
           {
           values[0] = 27.0/2.0*r;
           }
           else if (r < 3.0/9.0)
           {
           values[0] = -72.0 + 3123.0/2.0*r - 13122.0*pow(r,2) + 54675.0*pow(r,3) - 111537.0*pow(r,4) + 177147.0/2.0*pow(r,5);
           }
           else if (r < 6.0/9.0)
           {
           values[0] = 9.0 - 18.0*r;
           }
           else if (r < 7.0/9.0)
           {
           values[0] = -20079.0 + 136062.0*r - 367416.0*pow(r,2) + 494262.0*pow(r,3) - 662661.0/2.0*pow(r,4) + 177147.0/2.0*pow(r,5);
           }
           else if (r < 1)
           {
           values[0] = -27.0/2.0 + 27.0/2.0*r;
           }
           else
           {
           values[0] = 0;
           }
        }
        };
        '''
        
        popt_expr = Expression(expression_popt, degree=1)
        popt = interpolate(popt_expr,V)

        # initialize laplace of optimal adjoint state
        expression_laplacepopt ='''
        class my_expression : public Expression
        {
        public:
        void eval(Array<double>& values, const Array<double>& x) const
        {
            double r = sqrt( pow(x[0],2) + pow(x[1],2) );
            if (r < 2.0/27.0)
            {
            values[0] = 531441.0/32.0*r*(8.0 + 2025.0*r*r - 256.0*r);
            }
            else if (r < 2.0/9.0)
            {
            values[0] = 27.0/(2.0*r);
            }
            else if (r < 3.0/9.0)
            {
            values[0] = 9.0/2.0*(-11664.0 + 347.0/r + 109350.0*r +  729.0*r*r*(-544.0 + 675.0*r));
            }
            else if (r < 6.0/9.0)
            {
            values[0] = -18.0/r;
            }
            else if (r < 7.0/9.0)
            {
            values[0] = 9.0/2.0*(-326592.0 + 30236.0/r + 988524.0*r +   729.0*r*r*(-1616.0 + 675.0*r));
            }
            else if (r < 1)
            {
            values[0] = 27.0/(2.0*r);
            }
            else
            {
            values[0] = 0;
            }
        }
        };
        '''
        laplacepopt_expr = Expression(expression_laplacepopt, degree=1)
        laplacepopt = interpolate(laplacepopt_expr,V)
        
        # initialize optimal state and its laplacian
        yopt = interpolate(Expression('sin(2*pi*(x[0]*x[0]  + x[1]*x[1]))', degree=1),V)
        laplaceyopt = interpolate(Expression('8*pi*( cos(2*pi*(x[0]*x[0]  +  x[1]*x[1])) - 2*pi*(x[0]*x[0]  + x[1]*x[1])*sin(2*pi*(x[0]*x[0]  +  x[1]*x[1]))    )', degree=1),V)
        
    # construct e_omega
    eom = Function(V);
    eom.vector()[:] = - laplaceyopt.vector()[:]
    eom.vector()[:] -= uopt.vector()[:]

    # construct desired state y_d
    solve(a==eom*v*dx,temp,bc)

    yd = Function(V)
    yd.vector()[:] -=  temp.vector()[:]
    yd.vector()[:] -= laplacepopt.vector()[:]
    yd.vector()[:] += yopt.vector()[:]

    uopt_fine = interpolate(uopt_expr,V_fine)
    u_fine    = Function(V_fine)

    yd_ass = assemble(yd*v*dx)

    bc.apply(yd_ass)

    # assemble some matrizes and vectors
    A = assemble_scipy(a,bc)         # stiffness matrix
    M = assemble_scipy(w*v*dx,bc)    # mass matrix

    u1_f = interpolate(Expression("dummy", dummy = u1,degree=1),V)
    u2_f = interpolate(Expression("dummy", dummy = u2,degree=1),V)
    u3_f = interpolate(Expression("dummy", dummy = u3,degree=1),V)
    u4_f = interpolate(Expression("dummy", dummy = u4,degree=1),V)
    u5_f = interpolate(Expression("dummy", dummy = u5,degree=1),V)
              
    # Start active set method
    iteration = 1
    gamma = 1.0
    while (iteration <= maxiter):
        print('Computing solution with gamma=%1.3e' % (gamma))
        difference = 1
        inner_iter = 0
        
        while ((difference > 0) & (inner_iter <= max_inner_iter)):
            ga = 1+2*gamma/alpha

            # build active sets
            Q1.vector()[:] = 0.0
            Q2.vector()[:] = 0.0
            Q3.vector()[:] = 0.0
            Q4.vector()[:] = 0.0
            Q5.vector()[:] = 0.0

            Q1.vector()[ p.vector() < alpha/2*(  ga*u1 + u2   )  ] = 1.0
            Q2.vector()[ ( alpha/2*( u1 + ga*u2) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u2 + u3)   )     ] = 1.0
            Q3.vector()[ ( alpha/2*( u2 + ga*u3) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u3 + u4)   )     ] = 1.0
            Q4.vector()[ ( alpha/2*( u3 + ga*u4) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u4 + u5)   )     ] = 1.0
            Q5.vector()[   alpha/2*( u4 + ga*u5) < p.vector()  ] = 1.0

            # build "inactive" sets
            Q1_2.vector()[:] = 0.0
            Q2_3.vector()[:] = 0.0
            Q3_4.vector()[:] = 0.0
            Q4_5.vector()[:] = 0.0

            Q1_2.vector()[  (  alpha/2*( ga*u1 + u2 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u1 + ga*u2)   )   ] = 1.0
            Q2_3.vector()[  (  alpha/2*( ga*u2 + u3 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u2 + ga*u3)   )   ] = 1.0
            Q3_4.vector()[  (  alpha/2*( ga*u3 + u4 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u3 + ga*u4)   )   ] = 1.0
            Q4_5.vector()[  (  alpha/2*( ga*u4 + u5 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u4 + ga*u5)   )   ] = 1.0

            # build matrices

            # transform active set vectors to sparse diagonal matrizes
            Q1_m =  sp.spdiags(Q1.vector(),0,n,n,format='csr')   
            Q2_m =  sp.spdiags(Q2.vector(),0,n,n,format='csr') 
            Q3_m =  sp.spdiags(Q3.vector(),0,n,n,format='csr') 
            Q4_m =  sp.spdiags(Q4.vector(),0,n,n,format='csr') 
            Q5_m =  sp.spdiags(Q5.vector(),0,n,n,format='csr') 

            # transform "inactive" set vectors to sparse diagonal matrizes
            Q1_2m =  sp.spdiags(Q1_2.vector(),0,n,n,format='csr')   
            Q2_3m =  sp.spdiags(Q2_3.vector(),0,n,n,format='csr') 
            Q3_4m =  sp.spdiags(Q3_4.vector(),0,n,n,format='csr') 
            Q4_5m =  sp.spdiags(Q4_5.vector(),0,n,n,format='csr') 

            # compute the sum
            Qs1 = Q1_m + Q2_m + Q3_m + Q4_m + Q5_m
            Qs2 = Q1_2m + Q2_3m + Q3_4m + Q4_5m

            # build matrix
            # 1. line: Ay - Mu = 0
            # 2. line: Ap + My = My_d
            # 3. line: p + gamma*u + alpha*zeta = 0
            # 4. line: for subgradient and relation between u and Q_i see paper
            dF = sp.bmat( [ [-M, A, None, None], [None, M, A, None], [gamma*I, None, -I, alpha*I], [(I-Qs2)*I, None, None, (I-Qs1)*I]  ]    ,format='csr')

            # Assemble right hand side
            rhs_zeta.vector()[:] = Q1_m*u1_f.vector() + Q2_m*u2_f.vector() + Q3_m*u3_f.vector() + Q4_m*u4_f.vector() + Q5_m*u5_f.vector() + 0.5*Q1_2m*(u1_f.vector() + u2_f.vector()) + 0.5*Q2_3m*(u2_f.vector() + u3_f.vector()) + 0.5*Q3_4m*(u3_f.vector() + u4_f.vector()) + 0.5*Q4_5m*(u4_f.vector() + u5_f.vector())

            rhs = np.hstack([zero_vector.vector(), yd_ass, zero_vector.vector(), rhs_zeta.vector()])

            # Solve the linear system of equations
            zup = spsolve(dF,rhs)
            
            # split solution to get control, state, adjoint state and the subgradient
            u.vector()[:],y.vector()[:],p.vector()[:],zeta.vector()[:] = np.split(zup,4)

            # recompute active and inactive sets and compute the difference to the old sets
            difference = 0

            temp.vector()[:] = 0.0
            temp.vector()[ p.vector() < alpha/2*(  ga*u1 + u2   )  ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q1.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[ ( alpha/2*( u1 + ga*u2) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u2 + u3)   )     ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q2.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[ ( alpha/2*( u2 + ga*u3) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u3 + u4)   )     ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q3.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[ ( alpha/2*( u3 + ga*u4) < p.vector()   )   &   ( p.vector() < alpha/2*( ga*u4 + u5)   )     ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q4.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[   alpha/2*( u4 + ga*u5) < p.vector()  ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q5.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[  (  alpha/2*( ga*u1 + u2 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u1 + ga*u2)   )   ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q1_2.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[  (  alpha/2*( ga*u2 + u3 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u2 + ga*u3)   )   ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q2_3.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[  (  alpha/2*( ga*u3 + u4 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u3 + ga*u4)   )   ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q3_4.vector()))

            temp.vector()[:] = 0.0
            temp.vector()[  (  alpha/2*( ga*u4 + u5 ) <= p.vector() ) & (  p.vector() <= alpha/2*(u4 + ga*u5)   )   ] = 1.0
            difference = difference + sum(np.absolute(temp.vector()-Q4_5.vector()))

            print("change in active sets: %d" % difference)
            inner_iter = inner_iter+1


        # Compute L^2 Error
        u.set_allow_extrapolation(True)
        u_fine = project(u,V_fine)
        regerr = assemble( (u_fine - uopt_fine)**2*dx )
        
        err_gamma.append(gamma)
        err_reg.append(regerr)
        
        
        # increase iteration number and decrease gamma
        gamma = gamma/2.0
        iteration = iteration +1

        
    # Save the computed result on the finest grid
    file_control = File("control.pvd")
    file_state = File("state.pvd")
    file_adjoint = File("adjoint.pvd")

    file_control << u
    file_state << y
    file_adjoint << p
        
        
    # print the L^2 error
    print("L^2 error:")
    for k in range(0,len(err_gamma)):
        print "(", err_gamma[k], ",", err_reg[k], ")"
        
    # compute and print the numerical rate of convergence
    print("numerical convergence rate:")
    for k in range(1,len(err_gamma)):
        print np.log(err_reg[k-1]/err_reg[k])/np.log(2)
        
