#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 09:36:28 2023

@author: marcin
"""

import os.path
from PIL import Image

import itertools

import matplotlib.pyplot as plt
import torch as pt
from torch import matrix_exp as expm
from torch.linalg import eigh as eigh
import numpy as np
import pandas as pd
from numpy import linalg as LA

import sys
import time


 
id_local = pt.tensor([[1.,0],[0,1.]])
sigma_x = pt.tensor([[0,1.],[1.,0]])
sigma_y = 1j*pt.tensor([[0,-1.],[1.,0]])
sigma_z = pt.tensor([[1.,0],[0,-1.]])

hadamard = 1.0/pt.sqrt(pt.tensor(2))*pt.tensor([[1,1],[1,-1]])+1j*0

def get_string_operator(A, L, i):
    if(L == 1):
        return A
    else:
        Op = A
        if(i == 1):
            Id = id_local
            for i in range(2,L):
                Id = pt.kron(Id,id_local)
            Op = pt.kron(A,Id)
            return Op
    
        if(i == L):
            Id = id_local
            for j in range(2,L):
                Id = pt.kron(Id,id_local)
            Op = pt.kron(Id,A)
            return Op
    
        if(i>1 and i<L):
            counter_left = 1
            Id_left = id_local
            while(counter_left < i-1):
                Id_left = pt.kron(Id_left,id_local)
                counter_left = counter_left + 1
    
            counter_right = i+1
            Id_right = id_local
            while(counter_right < L):
                Id_right = pt.kron(Id_right,id_local)
                counter_right = counter_right + 1
            Op = pt.kron(Id_left, pt.kron(A, Id_right))
            return Op


def get_Identity(k):  # returns k-tensor product of the identity operator, ie. Id^k
    Id = id_local
    for i in range(1, k):
        Id = pt.kron(Id, id_local)
    return Id

 



from scipy.optimize import basinhopping

def get_expectation_value(psi,A):
    exp_val = pt.vdot(psi, A@psi)
    return exp_val.real
 


def generate_basis_via_sigma_z(L):
    D= 2**L
    basis = pt.zeros((2**L,L))
    for v in range(0,basis.shape[0]):
        fock_state = pt.zeros(D)
        fock_state[v] = 1
        for i in range(1,L+1):
            basis[v,i-1] = get_expectation_value(fock_state, Z[i])

    return (basis+1)/2

 
  
L_vec = [2, 3, 4, 5, 6]

Hamiltonian_type_vec = ["OAT", "TACT", "XXZ"]
for L in L_vec:
    D = 2**L
    Id = get_string_operator(id_local, L, 1)
    X = {}
    Y = {}
    Z = {}
    for i in range(1,L+1):
        index = i
        X[index] = get_string_operator(sigma_x, L, i)
        Y[index] = get_string_operator(sigma_y, L, i)
        Z[index] = get_string_operator(sigma_z, L, i)

    Sx_total = 0.5*sum([X[i] for i in range(1, L+1)])
    Sy_total = 0.5*sum([Y[i] for i in range(1, L+1)])
    Sz_total = 0.5*sum([Z[i] for i in range(1, L+1)])

    basis = generate_basis_via_sigma_z(L)
 

    #####################################################################################################        
    # Prepare set of coherent states for calculating Hussimi Q function
    psi_up = pt.zeros((D,)) + 0*1j
    psi_up[0] = 1
    N_grid = 200
    theta_vec = np.linspace(0, 1, N_grid)
    phi_vec = np.linspace(-1, 1, N_grid)
    psi_coherent = {}
    for idx_theta, theta in enumerate(theta_vec):
        for idx_phi, phi in enumerate(phi_vec):                
            psi_coherent[theta, phi] = expm(-1j*phi*np.pi*Sz_total)@expm(-1j*theta*np.pi*Sx_total)@psi_up                
    #####################################################################################################    

    #####################################################################################################
    # Prepare initial state
    phi = 0
    theta = -0.5
    psi_SCS = expm(-1j*phi*np.pi*Sz_total)@expm(-1j*theta*np.pi*Sy_total)@psi_up 
    #####################################################################################################
    
    t_max = 2*pt.pi
    dt = .05
    N_t = int(t_max/dt)

    for Hamiltonian_type in Hamiltonian_type_vec:
        df_evolution = []
        
        if(Hamiltonian_type == "OAT"):
            H = Sz_total@Sz_total

        if(Hamiltonian_type == "TACT"):
            H = Sz_total@Sz_total - Sy_total@Sy_total

        if(Hamiltonian_type == "XXZ"):
            delta = 0.5
            H = sum([ (X[i]@X[i+1] + Y[i]@Y[i+1] + delta*Z[i]@Z[i+1]) for i in range(1,L)])
        
        # Construct evolution operator |psi(t+dt)> = U|psi(t)>
        U_evolution_dt = expm(-1j*dt*H)


        #########################################
        ###             Time evolution    #######          
        
        
        # Initialize spins state 
        psi_t = psi_SCS
        for t_idx in range(0, N_t):
    
            t = t_idx*dt
            t_over_pi = t/np.pi
    
            psi_t = U_evolution_dt@psi_t
            norm = pt.sum(pt.abs(psi_t)**2.0)
            
            ###############################################################
            # Calculate Hussimi Function
            Q_hussimi = np.zeros((theta_vec.shape[0], phi_vec.shape[0]))
            Bloch_sphere_representation = []
            for idx_theta, theta in enumerate(theta_vec):
                for idx_phi, phi in enumerate(phi_vec):                
                    tmp_ = np.vdot(psi_coherent[theta, phi],  psi_t)
                    Q_huss = np.abs(tmp_)**2  
                    Q_hussimi[idx_theta, idx_phi] = Q_huss
                    
                    # from spherical to cartezian coordinates
                    r = 1
                    x = r*np.sin(theta*np.pi)*np.cos(phi*np.pi)
                    y = r*np.sin(theta*np.pi)*np.sin(phi*np.pi)
                    z = r*np.cos(np.pi*theta)
                    Bloch_sphere_representation.append([x, y, z, Q_huss])
            
            Bloch_sphere_representation = np.array(Bloch_sphere_representation)
            #############################################################

            dict_t = {
                        "L"                 : L,
                        "Hamiltonian"       : Hamiltonian_type,
                        "t_i"               : t_idx,
                        "t/pi"              : t_over_pi,
                        "norm_t"            : norm,
                        "Q_hussimi_t"       : Q_hussimi,
                        "Bloch_sphere_t"    : Bloch_sphere_representation,
                        "phi_vec"           : phi_vec,
                        "theta_vec"         : theta_vec,
                        }

            df_evolution.append(dict_t)                            
            string = " L = " + str(L)  + " | " + Hamiltonian_type
            string = string + " | t/pi = " + "{:2.4f}".format(t_over_pi)  + " | {:d}/{:d}".format(t_idx + 1, N_t)
            print(string)
            
        df_evolution = pd.DataFrame(df_evolution)             
        path = './'
        filename_ = "data_" + Hamiltonian_type + "_L." + str(L) + ".pkl"
        filename = path + filename_
        pd.to_pickle(df_evolution, filename)
 
