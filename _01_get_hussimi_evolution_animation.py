#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marcin Plodzien
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
 
L_vec = [2, 3, 4, 5, 6]

duration = 600 # frame duration in [ms]
frame_delta = 2
for L in L_vec:
    #% Hussimi function plot
    #
    FontSize = 20
    Hamiltonian_type_vec_ = ["OAT", "TACT", "XXZ"]
    # Hamiltonian_type_vec_ = ["OAT"]
    fig, ax = plt.subplots(1,1, figsize = (14, 7))
    
    for idx, Hamiltonian_type in enumerate(Hamiltonian_type_vec_):
        
        path = './'
        filename_ = "data_" + Hamiltonian_type + "_L." + str(L) + ".pkl"
        filename = path + filename_
        df_evolution = pd.read_pickle(filename)
        
        
        
        theta_vec = df_evolution["theta_vec"][0]
        phi_vec = df_evolution["phi_vec"][0]
        
        
            
        cond_H_type = df_evolution["Hamiltonian"] == Hamiltonian_type
        data_ = df_evolution[cond_H_type].iloc[::frame_delta,:]
        data_ = data_.reset_index()
        Q_list = data_["Q_hussimi_t"].values # Replace with your actual list of Husimi functions
        
    
        path_gif = './'
        filename_animation = "fig_Hussimi_function_evolution_" + Hamiltonian_type + "_L." + str(L) + ".gif"
        filename = path_gif + filename_animation
        images = []  # List to store frames as images
        
        # Loop over each time step and create a frame
        for k, Q in enumerate(Q_list):
            ax.clear()  # Clear the previous plot
            t_over_pi = data_["t/pi"][k] 
            cax = ax.imshow(Q, extent=[phi_vec[0], phi_vec[-1], theta_vec[0], theta_vec[-1]],
                            aspect=0.75, origin='lower', cmap='viridis')
            ax.set_xlabel(r"$\phi/\pi$", fontsize = FontSize)
            ax.set_ylabel(r"$\theta/\pi$", fontsize = FontSize)
            
            title_string = r"$|\theta,\phi\rangle = e^{-i \phi \hat{S}_z}e^{-i\theta \hat{S}_x}|\uparrow\rangle^{\otimes L}$"  
            title_string = title_string + r", $|\psi_t\rangle = e^{-i t \hat{H}}|-\pi/2,0\rangle$"
            title_string = title_string + r", $Q_t(\phi,\theta) = |\langle \theta,\phi|\psi_t\rangle|^2$" + "\n"
            title_string = title_string + Hamiltonian_type + " | L = " + str(L) + r" | $ k = {:d} | t  = {:2.3f} \times \pi$".format(k, t_over_pi)
             
            
            
            ax.set_title(title_string, fontsize = FontSize)
            ax.tick_params(axis='x', labelsize=FontSize)
            ax.tick_params(axis='y', labelsize=FontSize)
            
            # Draw the plot but do not show it
            fig.canvas.draw()
        
            # Convert the plot to an image and append to the list
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(Image.fromarray(image))
        
        # Save all images as an animated GIF
        images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
        
        # Close the plot to free resources
        plt.close()
        print(f"Animated GIF saved as {filename}")
        
        
    #% Hussimi on Bloch sphere
    #
    FontSize = 20
    # Hamiltonian_type_vec_ = ["OAT", "TACT", "XXZ"]
    fig, ax = plt.subplots(1,1, figsize = (14, 14) )
    for idx, Hamiltonian_type in enumerate(Hamiltonian_type_vec_):
        
        path = './'
        filename_ = "data_" + Hamiltonian_type + "_L." + str(L) + ".pkl"
        filename = path + filename_
        df_evolution = pd.read_pickle(filename)
        theta_vec = df_evolution["theta_vec"][0]
        phi_vec = df_evolution["phi_vec"][0]
        
        cond_H_type = df_evolution["Hamiltonian"] == Hamiltonian_type
        data_ = df_evolution[cond_H_type].iloc[::frame_delta,:]
        data_ = df_evolution[cond_H_type]
        data_ = data_.reset_index()
        Bloch_sphere_list = data_["Bloch_sphere_t"].values # Replace with your actual list of Husimi functions
        
    
        path_gif = './'
        filename_animation = "fig_Hussimi_function_evolution_Bloch_sphere_" + Hamiltonian_type + "_L." + str(L) + ".gif"
        filename = path_gif + filename_animation
        images = []  # List to store frames as images
        
        
        # Loop over each time step and create a frame
        for k, Bloch_sphere in enumerate(Bloch_sphere_list):
            
            ax.clear()  # Clear the previous plot
            ax = fig.add_subplot(111, projection='3d')
            t_over_pi = data_["t/pi"][k] 
    
            X = Bloch_sphere[:,0]                        
            Y = Bloch_sphere[:,1]
            Z = Bloch_sphere[:,2]
            Q_huss = Bloch_sphere[:,3]
            
       
            idx = np.where( np.abs(Q_huss - np.max(Q_huss))<0.2)[0]
            
            X_ = X[idx]
            Y_ = Y[idx]
            Z_ = Z[idx]
            c_ = Q_huss[idx]
            
            cax = ax.scatter(X_, Y_, Z_, c  = c_, cmap='gnuplot')
            ax.set_xlim([-1, 1])  # Example: set x-axis limits from -1 to 1
            ax.set_ylim([-1, 1])  # Example: set y-axis limits from -1 to 1
            ax.set_zlim([-1, 1])  # Example: set z-axis limits from -1 to 1
    
            
            title_string = r"$|\theta,\phi\rangle = e^{-i \phi \hat{S}_z}e^{-i\theta \hat{S}_x}|\uparrow\rangle^{\otimes L}$"  
            title_string = title_string + r", $|\psi_t\rangle = e^{-i t \hat{H}}|-\pi/2,0\rangle$"
            title_string = title_string + r", $Q_t(\phi,\theta) = |\langle \theta,\phi|\psi_t\rangle|^2$" + "\n"
            title_string = title_string + Hamiltonian_type + " | L = " + str(L) + r" | $ k = {:d} | t  = {:2.3f} \times \pi$".format(k, t_over_pi)
             
            
            
            ax.set_title(title_string, fontsize = FontSize)
            ax.tick_params(axis='x', labelsize=FontSize)
            ax.tick_params(axis='y', labelsize=FontSize)
            
            # Draw the plot but do not show it
            fig.canvas.draw()
        
            # Convert the plot to an image and append to the list
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(Image.fromarray(image))
        
        # Save all images as an animated GIF
        images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)
        
        # Close the plot to free resources
        plt.close()
        print(f"Animated GIF saved as {filename}")    
