#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
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

n_update, volt, temperature, nsec, ntimestep_write, platform_name, ResidueConnectivityFiles, FF_files, grp_c, grp_d, grp_neu, functional_grp, redox_mol, redox_state_f_xml, redox_states, E_fermi, EA_neu, EA_red1 = read_input(args.input)

if n_update is not None:
    outPath = 'simmd_' + n_update + "step_" + volt + "V_" + nsec + "ns_" + temperature + 'K'
else:
    outPath = "output" + strftime("%s",gmtime())

if os.path.exists(outPath):
    shutil.rmtree(outPath)

#strdir ='../'
os.mkdir(outPath)
os.chdir(outPath)
chargesFile = open("charges.dat", "w")
print(outPath)

pdb = args.pdb
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
sim.exlusionNonbondedForce(sim.electrode.all_cathode_atomindices, sim.electrode.all_anode_atomindices)
for i_redox in range(len(sim.redox_molecules)):
    sim.exlusionNonbondedForce2(sim.redox_molecules[i_redox].atomindex, sim.electrode.neutral)
sim.exlusionNonbondedForce1(sim.graph)
sim.exlusionNonbondedForce2(sim.grpc, sim.electrode.neutral)
sim.exlusionNonbondedForce2(sim.electrode.dummy, sim.electrode.neutral)
sim.simmd.context.reinitialize()
sim.simmd.context.setPositions(sim.initialPositions)
sim.initialize_energy()
sim.equilibration()

cell_dist, z_L, z_R = Distance(sim.electrode.c562_1, sim.electrode.c562_2, sim.initialPositions)
print(z_L, z_R)
print('cathode-anode distance (nm)', cell_dist)
boxVecs = sim.simmd.topology.getPeriodicBoxVectors()
crossBox = numpy.cross(boxVecs[0], boxVecs[1])
sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2
print(sheet_area)


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
E_fermi = float(E_fermi) *96.487 # convert eV to kj/mol
EA_neu = float(EA_neu) * 96.487
EA_red1 = float(EA_red1) * 96.487


sim.initializeCharge( Ngraphene_atoms, sim.graph, area_atom, Voltage, Lgap, conv, small, cell_dist)

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

    if redox_mol == "None":
        pass
    else:
        state = sim.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
        PE_old = state.getPotentialEnergy()
        redox_mol_i = random.randint(0, len(sim.redox_molecules)-1)
        init_redox_atomcharges, init_redoxstate_list = get_charges_redoxstates_from_atomindex(sim.redox_molecules_atomindices, sim.nbondedForce)
        electron_transfer = ['oxidation', 'reduction']
        ET_i = random.choice(electron_transfer)
        if ET_i == 'oxidation':
            sim.redox_molecules[redox_mol_i].oxidation(sim.nbondedForce, redox_states, init_redoxstate_list[redox_mol_i], redox_mol_i, sim.Nredox_cathode)
        if ET_i == 'reduction':
            sim.redox_molecules[redox_mol_i].reduction(sim.nbondedForce, redox_states, init_redoxstate_list[redox_mol_i], redox_mol_i, sim.Nredox_cathode)
        if sim.redox_molecules[redox_mol_i].flag == 1:
            ntrials = 0
            naccept = 0
            sim.MonteCarlo_redox(ntrials, naccept, Ngraphene_atoms, Niter_max, sim.graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol, redox_mol_i, redox_states, ET_i, PE_old._value, E_fermi, EA_neu, EA_red1)

            new_redox_atomcharges, new_redoxstate_list = get_charges_redoxstates_from_atomindex(sim.redox_molecules_atomindices, sim.nbondedForce)
            #print("redox states before monte carlo\n", init_redoxstate_list)
            print("redox states (cathode) after monte carlo\n", new_redoxstate_list[:sim.Nredox_cathode])
            print("redox states (anode) after monte carlo\n", new_redoxstate_list[sim.Nredox_cathode:])
            #list_reductions = [ i for i in range(len(new_redoxstate_list)) if new_redoxstate_list[i] < sim.redox_molecules_redoxstates[0] ]
            #list_oxidations = [ i for i in range(len(new_redoxstate_list)) if new_redoxstate_list[i] > sim.redox_molecules_redoxstates[0] ]
            #print("N_reduction:", list_reductions, len(list_reductions)) 
            #print("N_oxdation", list_oxidations, len(list_oxidations))


print('Done!')

exit()
