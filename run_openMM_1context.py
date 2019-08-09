#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
#import simtk.unit as unit
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
from copy import deepcopy
import os
import sys
import numpy
import argparse
import shutil

from subroutines_1context import *

parser = argparse.ArgumentParser()
parser.add_argument("pdb", type=str, help="PDB file with initial positions")
parser.add_argument("input", type=str, help="input parameters")
args = parser.parse_args()

n_update, volt, temperature, nsec, ntimestep_write, platform_name, ResidueConnectivityFiles, FF_files, grp_c, grp_d, grp_neu, functional_grp, redox_mol, redox_state_f_xml, et_electrode = read_input(args.input)

if n_update is not None:
    outPath = 'simmd_' + n_update + "step_" + volt + "V_" + nsec + "ns_" + temperature + 'K'
else:
    outPath = "output" + strftime("%s",gmtime())

if os.path.exists(outPath):
    shutil.rmtree(outPath)

#strdir ='../'
#forcefieldfolder = '../ffdir/'
os.mkdir(outPath)
os.chdir(outPath)
chargesFile = open("charges.dat", "w")
print(outPath)

pdb = args.pdb
#device_idx = args.devices
sim = MDsimulation( 
        pdb, 
        float(temperature),
        ntimestep_write,
        platform_name,
        ResidueConnectivityFiles, 
        FF_files,
        grp_c, grp_d, grp_neu,
        functional_grp, redox_mol,
        redox_state_f_xml
)

# add exclusions for intra-sheet non-bonded interactions.
sim.exlusionNonbondedForce(sim.electrode_1_arr, sim.electrode_2_arr)
sim.exlusionNonbondedForce2(sim.grpc, sim.grph)
sim.exlusionNonbondedForce2(sim.grp_dummy, sim.grph)
sim.exlusionNonbondedForce2(sim.grpc, sim.grp_dummy)
sim.simmd.context.reinitialize()
sim.simmd.context.setPositions(sim.initialPositions)
sim.initialize_energy()
sim.equilibration()

cell_dist, z_L, z_R = Distance(sim.c562_1, sim.c562_2, sim.initialPositions)
print(z_L, z_R)
print('cathode-anode distance (nm)', cell_dist)
boxVecs = sim.simmd.topology.getPeriodicBoxVectors()
crossBox = numpy.cross(boxVecs[0], boxVecs[1])
sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2
print(sheet_area)

#sim.Get_redox_charge_array("../ffdir/charge_qsem.xml")

print('Starting Production NPT Simulation...')


#************ get rid of the MD loop, just calculating converged charges ***********
Ngraphene_atoms = len(sim.graph)

# one sheet here
area_atom = sheet_area / (Ngraphene_atoms / 2) # this is the surface area per graphene atom in nm^2
conv = 18.8973 / 2625.5  # bohr/nm * au/(kJ/mol)
# z box coordinate (nm)
zbox=boxVecs[2][2] / nanometer
Lgap = (zbox - cell_dist) # length of vacuum gap in nanometers, set by choice of simulation box (z-dimension)
print('length of vacuum gap (nm)', Lgap)
Niter_max = 100  # maximum steps for convergence
#tol=0.01 # tolerance for average deviation of charges between iteration
kb = BOLTZMANN_CONSTANT_kB._value * AVOGADRO_CONSTANT_NA._value * (1/1000) # convert to kJ/(mol*K)
tol = kb * float(temperature)  # convergence threshhod for deviation of electrostatic between iteration (kBT in kJ/mol)
Voltage = float(volt)  # external voltage in Volts
Voltage = Voltage * 96.487  # convert eV to kJ/mol to work in kJ/mol
q_max = 2.0  # Don't allow charges bigger than this, no physical reason why they should be this big
f_iter = int(( float(nsec) * 1000000 / int(n_update) )) + 1  # number of iterations for charge equilibration
#print('number of iterations', f_iter)
small = 1e-4

sim.initializeCharge( Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, small, cell_dist)
if et_electrode == 'cathode':
    redox_charges_i_current = deepcopy(sim.redoxcharges_state1_cathode)
elif et_electrode == 'anode':
    redox_charges_i_current = deepcopy(sim.redoxcharges_state1_anode)

for i in range(1, f_iter ):
    print()
    print(i,datetime.now())

    sim.simmd.step( int(n_update) )

    state = sim.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))

    positions = state.getPositions()
    #sim.Charge_solver( Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol )
    sim.ConvergedCharge( Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol )
    #sim.ConvergedCharge( Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max )
    #sumq_cathode, sumq_anode = sim.FinalCharge(Ngraphene_atoms, sim.graph, args, i, chargesFile)
    #print( 'total charge on graphene (cathode,anode):', sumq_cathode, sumq_anode )
    #print('Charges converged, Energies from full Force Field')
    #sim.PrintFinalEnergies()

    #ind_Q = get_Efield(sim.solvent_list)
    #ana_Q_Cat, ana_Q_An = ind_Q.induced_q( z_L, z_R, cell_dist, sim, positions, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv)
    #print('Analytical Q_Cat, Q_An :', ana_Q_Cat, ana_Q_An)    
    #sim.Scale_charge( Ngraphene_atoms, sim.graph, ana_Q_Cat, ana_Q_An, sumq_cathode, sumq_anode)
    ntrials = 0
    naccept = 0
    if et_electrode == 'cathode':
        #redox_charges_i_new = sim.Swap_redox_ff( sim.redox_atomindex_electrode_1, sim.redox_charges_newarray1, sim.redox_charges_oldarray1, ntrials, naccept, Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, redox_charges_i_current, tol )
        redox_charges_i_new = sim.Swap_redox_ff( sim.redox_atomindices_cathode, sim.redoxcharges_state2_cathode, sim.redoxcharges_state1_cathode, ntrials, naccept, Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, redox_charges_i_current, tol )
        redox_charges_i_current = redox_charges_i_new
    elif et_electrode == 'anode':
        #redox_charges_i_new = sim.Swap_redox_ff( sim.redox_atomindex_electrode_2, sim.redox_charges_newarray2, sim.redox_charges_oldarray2, ntrials, naccept, Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, redox_charges_i_current, tol)
        redox_charges_i_new = sim.Swap_redox_ff( sim.redox_atomindices_anode, sim.redoxcharges_state2_anode, sim.redoxcharges_state1_anode, ntrials, naccept, Niter_max, Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, redox_charges_i_current, tol)
        redox_charges_i_current = redox_charges_i_new

print('Done!')

exit()
