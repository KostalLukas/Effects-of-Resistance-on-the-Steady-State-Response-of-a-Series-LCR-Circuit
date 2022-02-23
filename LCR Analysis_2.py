'''
LCR Circuit Analysis 2

Lukas Kostal, 23.2.2022, ICL
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp
import scipy.optimize as op

#set fonts and font size for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.size'] = 10

#define fitting function for amplitude plot
def ampli(f, N, R, L ,C):
    omg = 2*sp.pi*f
    A = N/np.sqrt((omg**2 - 1/(L*C))**2 + (R*omg/L)**2)
    return(A)

#define fitting function for phase difference
def phase(f, R, L, C):
    omg = 2 * sp.pi * (f)
    phi = np.arctan(R * (omg) / (L * (omg) ** 2 - 1 / C)) + sp.pi
    return(np.fmod(phi, sp.pi) - sp.pi)

#define function to calcualte phase from simualted data
def simphase(Re, Im):
    phi = np.arctan(Im/Re)
    for j in range (0, len(Re)):
        if phi[j] > 0:
            phi[j] = phi[j] - np.pi
    return(phi)

#namig of datasets to be imported
Ri = ['R1_2', 'R2_1', 'R3_1']

#input values of components and their approximated uncertainties
L = 1e-3 #inductance of inductor in H
R_L = 3.63 #series resistance of inductor in Ohm
Lerr = 1e-4 #expected uncertainty in inductance in H
C = 1e-7 #capacitance of capacitor in F
Cerr = 1e-8 #expected uncertainty in capacitance in F
R = np.array([1.00, 2.04, 3.06]) #resistance of the 3 resistors in Ohm
Rerr = 0.01 #expected uncertainty in resistance in Ohm

Rtot = R + R_L

#arrays to store fitting parameters
simamp = np.empty(3, object)
simphi = np.empty(3, object)
thamp = np.empty(3, object)
thphi = np.empty(3, object)
amp = np.empty(3, object)
amp_std = np.empty(2, object)
phi = np.empty(3, object)
phi_std = np.empty(2, object)

#loop over all of the resistor values
for i in range(0,len(Ri)):
    
    #load data from .csv files units are kHz, kHz, Hz, V, uV, °, °
    f_target, f, f_std, VCpp, VCpp_std, phs, phs_std = np.loadtxt(f'Data/{Ri[i]}.csv', skiprows=(1),delimiter=',', unpack=True)
    
    #convert peak to peak voltage to voltage
    VC = VCpp/2
    #covert voltage standard deviation from uV peak to peak to V 
    VC_std = VCpp_std * 1e-6 /2
    #covert f from kHz to Hz
    f = f * 1e3
    #covert phase difference and associated uncertainty from deg to rad and offset 
    phs = sp.pi * (-phs/180)
    phs_std = sp.pi * phs_std/180
    
    #load simulated data from .csv file
    #units are Hz, V, V respectively
    f_sim, V_re, V_im = np.loadtxt(f'Simulation/R{i+1}.csv', skiprows=(1),delimiter=',', unpack=True)
    
    #calcualte simulated amplitude as the absolute value
    V_sim = np.sqrt(np.square(V_re) + np.square(V_im))
    #scale the simulated amplitude
    V_sim = np.amax(VC)/np.amax(V_sim) * V_sim 
    #calcualte the simulated phase as the argument
    phs_sim = simphase(V_re, V_im)
    
    #set resistance as a fixed parameter
    Rampli = lambda f, N, L, C: ampli(f, N, Rtot[i], L, C)
    Rphase = lambda f, L, C: phase(f, Rtot[i], L, C)

    #initial guess for amplitude simulation curve fit
    Vmax = np.amax(V_sim)
    N_ig = Vmax * Rtot[i]/np.sqrt(L**3 * C)
    amp_ig= np.array([N_ig, L, C])
    #curve fit for the amplitude simulation
    amp_sim, nothing = op.curve_fit(Rampli, f_sim, V_sim, amp_ig)

    #initial guess for amplitude curve fit
    Vmax = np.amax(VC)
    N_ig = Vmax * Rtot[i]/np.sqrt(L**3 * C)
    amp_ig= np.array([N_ig, L, C])
    # curve fit for the amplitude data
    amp_op, amp_cov = op.curve_fit(Rampli, f, VC, amp_ig, sigma=VC_std, absolute_sigma=True)

    #initial guess for phase simulation curve fit
    phi_ig = np.array([L, C])
    # curve fit for the phase simulation
    phi_sim, nothing = op.curve_fit(Rphase, f_sim, phs_sim, phi_ig)

    #initial guess for phase curve fit
    phi_ig = np.array([L, C])
    # curve fit for the phase data
    phi_op, phi_cov = op.curve_fit(Rphase, f, phs, phi_ig, sigma=phs_std, absolute_sigma=True)
 
    #define array of frequencies for plotting
    fplot = np.linspace(8000, 25000, 500)   
 
    #setting the plot title and labling axis
    plt.title('$V_{C}$' f'Resonance Curve for {Ri[i]}')
    plt.ylabel('$V_C$ / V')
    plt.xlabel('$f$ / Hz')

    #plot the datapoints, fitted curves and predicted values
    plt.plot(fplot, Rampli(fplot, *amp_sim), color='forestgreen', label='simulation')
    plt.plot(fplot, Rampli(fplot, *amp_ig), color='darkorange', label='theoretical')
    plt.errorbar(f, VC, xerr=f_std, yerr=VC_std, fmt='x', color='black', capsize=3)
    plt.plot(f, Rampli(f, *amp_op), color='royalblue', label='experimental')
    plt.legend() #make a plot legend
    plt.savefig(f'Output/Ampli {Ri[i]}.png', bbox_inches='tight', dpi=200) #save the plot
    plt.show()
    
    #remove invalid points for phase plot for R1
    if i==0:
        f = f[:-3]
        f_std = f_std[:-3]
        phs = phs[:-3]
        phs_std = phs_std[:-3]
        
    #remove invalid points for phase plot for R2
    if i==1:
        f = f[:-2]
        f_std = f_std[:-2]
        phs = phs[:-2]
        phs_std = phs_std[:-2]
    
    #setting the plot title and labling axis
    plt.title(f'Phase Difference Curve {Ri[i]}')
    plt.ylabel('$\phi$ / rad')
    plt.xlabel('$f$ / Hz')

    #plot the datapoints, fitted curves and predicted values
    plt.plot(fplot, Rphase(fplot, *phi_sim), color='forestgreen', label='simulation')
    plt.plot(fplot, Rphase(fplot, *phi_ig), color='darkorange', label='theoretical')
    plt.errorbar(f, phs, xerr=f_std, yerr=phs_std, fmt='x', color='black', capsize=3)
    plt.plot(f, Rphase(f, *phi_op), color='royalblue', label='experimental')
    plt.legend() #make a plot legend
    plt.savefig(f'Output/Phase {Ri[i]}.png', bbox_inches='tight', dpi=200) #save the plot
    plt.show()
    
    #calculate theoretical values for resonant frequency and expected uncertainty
    res = 1/np.sqrt(L*C) /(2* np.pi * 1000)
    res_err = 1/2 * np.sqrt(Lerr**2 * L**-3 * C**-1 + Cerr**2 * L**-1 * C**-3) /(2* np.pi * 1000)
    
    #calculate theoretical values for quality factor and expected uncertainty
    Q = 1/Rtot[i] * np.sqrt(L/C)
    Q_err = 1/2 * np.sqrt( 4* Rerr**2 * Rtot[i]**-4 * L * C**-1
                          + Lerr**2 * Rtot[i]**-2 * L**-1 * C**-1 
                          + Cerr**2 * Rtot[i]**-2 * L * C**-1)
    
    #calculate LTspice simulation values
    res_sim = 1/(np.sqrt(amp_sim[1] * amp_sim[2]) * 2*sp.pi) /1000
    Q_sim = 1/Rtot[i] * np.sqrt(amp_sim[1]/amp_sim[2])
    
    
    #calculate experimental resonant frequency and expected uncertainty from amplitude fit
    res_amp = 1/(np.sqrt(amp_op[1] * amp_op[2]) * 2*sp.pi) /1000
    res_amp_err = 1/2 * np.sqrt(np.absolute(amp_cov[1,1] * amp_op[1]**-3 * amp_op[2]**-1 
                                + amp_cov[2,2] * amp_op[1]**-1 * amp_op[2]**-3)) /(2*sp.pi*1000)
    
    #calculate experimental quality factor and expected uncertainty from amplitude fit
    Q_amp = 1/Rtot[i] * np.sqrt(amp_op[1]/amp_op[2])
    Q_amp_err = 1/2 * np.sqrt(np.absolute( 4* Rerr**2 * Rtot[i]**-4 * amp_op[1] * amp_op[2]
                              + amp_cov[1,1] * Rtot[i]**-2 * amp_op[1]**-1 * amp_op[2]**-1
                              + amp_cov[2,2] * Rtot[i]**-2 * amp_op[1] * amp_op[2]**-3 ))
    
    #calculate experimental resonant frequency and expected uncertainty from phase fit
    res_phi = 1/(np.sqrt(phi_op[0]*phi_op[1]) * 2*sp.pi) /1000
    res_phi_err = 1/2 * np.sqrt(np.absolute(phi_cov[0,0] * phi_op[0]**-3 * phi_op[1]**-1 
                                + phi_cov[1,1] * phi_op[0]**-1 * phi_op[1]**-3)) /(2*sp.pi*1000)
    
    #calculate experimental quality factor and expected uncertainty from phase fit
    Q_phi = 1/Rtot[i] * np.sqrt(phi_op[0]/phi_op[1])
    Q_phi_err = 1/2 * np.sqrt(np.absolute( 4* Rerr**2 * Rtot[i]**-4 * phi_op[0] * phi_op[1]
                              + phi_cov[0,0] * Rtot[i]**-2 * phi_op[0]**-1 * phi_op[1]**-1
                              + amp_cov[1,1] * Rtot[i]**-2 * phi_op[0] * phi_op[1]**-3 ))
    
    #calculate percentage deviations from theoretical results
    res_amp_pd = np.absolute((res - res_amp)/res_amp)*100
    res_phi_pd = np.absolute((res - res_phi)/res_phi)*100 
    Q_amp_pd = np.absolute((Q - Q_amp)/Q_amp)*100
    Q_phi_pd = np.absolute((Q - Q_phi)/Q_phi)*100
    
    #print all of the results
    print('-----------------------------------------------------------------------')
    print('For', Ri[i], ' R= %.2f ± %.2f Ω' % (R[i], Rerr))
    print()
    print('Theoretical values:')
    print('f_0 = %.1f ± %.1f kHz (3sf) Q = %.1f ± %.1f (3sf)' % (res, res_err, Q, Q_err))
    print()
    print('LTspice simulation:')
    print('f_0 = %.1f kHz (3sf) Q = %.1f (3sf)' % (res_sim, Q_sim))
    print()
    print('Experimental values using amplitude:')
    print('f_0 = %.3f ± %.3f kHz (5sf) Q = %.3f ± %.3f (5sf)' % (res_amp, res_amp_err, Q_amp, Q_amp_err))
    print('Deviation from theoretical values:')
    print('f_0 deviation = %.1f %% (2sf) Q deviation = %.0f %% (3sf)' % (res_amp_pd, Q_amp_pd))
    print()
    print('Experimental values using phase difference:')
    print('f_0 = %.2f ± %.2f kHz (4sf) Q = %.1f ± %.1f (3sf)' % (res_phi, res_phi_err, Q_phi, Q_phi_err))
    print('Deviation from theoretical values:')
    print('f_0 deviation = %.1f %% (2sf) Q deviation = %.0f %% (3sf)' % (res_phi_pd, Q_phi_pd))
    print()
    print()

    #append resistance to the fitting parameters and save for fitting outside of loop
    amp_sim = np.insert(amp_sim, 1, Rtot[i], axis=0)
    phi_sim = np.insert(phi_sim, 0, Rtot[i], axis=0)
    amp_ig = np.insert(amp_ig, 1, Rtot[i], axis=0)
    phi_ig = np.insert(phi_ig, 0, Rtot[i], axis=0)
    amp_op = np.insert(amp_op, 1, Rtot[i], axis=0)
    phi_op = np.insert(phi_op, 0, Rtot[i], axis=0)
    simamp[i] = amp_sim
    simphi[i] = phi_sim
    thamp[i] = amp_ig
    thphi[i] = phi_ig
    amp[i] = amp_op
    phi[i] = phi_op
    
#set fonts and font size for plotting
plt.rcParams['font.family'] = 'baskerville'
plt.rcParams['mathtext.fontset'] = "cm"
plt.rcParams['font.size'] = 8
    
#define colors for plotting
colr = ['royalblue', 'limegreen', 'orangered']

#define array of frequencies for plotting
f = np.linspace(8000, 25000, 500)

#setup subplots
fig, axs = plt.subplots(3, sharex=True, sharey=True)

#setting the plot title and labling axis
plt.title('$V_{C}$ Resonance Curve Comaprison')
plt.ylabel('$V_C$ / V')
plt.xlabel('$f$ / Hz')

#loop for plotting the amplitude subplots
for i in range(0, 3):
    plt.xlabel('$f$ / Hz')
    axs[0].plot(f, ampli(f, *amp[i]), color=colr[i])
    #axs[0].axvline(x=res_amp)
    axs[0].set_title('Measured Resonance Curve')
    axs[1].plot(f, ampli(f, *thamp[i]), color=colr[i])
    axs[1].set_title('Theoretical Resonance, Curve')
    axs[2].plot(f, ampli(f, *simamp[i]), color=colr[i], label=f'R= {R[i]} $\\Omega$')
    axs[2].set_title('Simulated Resonance Curve')

#set subplot separation, create legend, save plot
fig.tight_layout(pad=0.2)
plt.legend()
plt.savefig('Output/Resonance Curve Comaprison.png', bbox_inches='tight', dpi=200) #save the plot
plt.show()

#setup subplots
fig, axs = plt.subplots(3, sharex=True, sharey=True)

#setting the plot title and labling axis
plt.title('Phase Difference Curve Comparison')
plt.ylabel('$\phi$ / rad')
plt.xlabel('$f$ / Hz')

#loop for plotting the phase subplots
for i in range(0, 3):
    plt.xlabel('$f$ / Hz')
    axs[0].plot(f, phase(f, *phi[i]), color=colr[i])
    axs[0].set_title('Measured Phase Curve')
    axs[1].plot(f, phase(f, *thphi[i]), color=colr[i])
    axs[1].set_title('Theoretical Phase Curve')
    axs[2].plot(f, phase(f, *simphi[i]), color=colr[i], label=f'R= {R[i]} $\\Omega$')
    axs[2].set_title('Simulated Phase Curve')
    
#set subplot separation, create legend, save plot
fig.tight_layout(pad=0.2)
plt.legend() #make a plot legend
plt.savefig('Output/Phase Difference Curve Comparison.png', bbox_inches='tight', dpi=200) #save the plot
plt.show()


#define array of frequencies for plotting phase zoomed on resonance
f = np.linspace(16000, 18000, 500)

#setting the plot title and labling axis
plt.title('Phase Difference Curve Comparison')
plt.ylabel('$\phi$ / rad')
plt.xlabel('$f$ / Hz')

#loop for plotting the phase graphs zoomed on resonance
for i in range(0, 3):
    plt.plot(f, phase(f, *phi[i]), color=colr[i], label=f'R= {R[i]} $\\Omega$')
    
plt.legend() #make a plot legend
plt.savefig('Output/Phase Difference Curve Comparison Closeup.png', dpi=200) #save the plot
plt.show()


