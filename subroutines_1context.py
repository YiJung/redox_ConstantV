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
import random
import argparse
import shutil

#from run_openMM_test_0816 import graph

class MDsimulation:
    #def __init__(self, read_pdb, index, temperature, ResidueConnectivityFiles, FF_files):
    #def __init__(self, read_pdb, temperature, ntimestep_write, platform_name, ResidueConnectivityFiles, FF_files, grp_c, grp_d, grp_neu, functional_grp, redox_mol):
    def __init__(self, read_pdb, temperature, ntimestep_write, platform_name, ResidueConnectivityFiles, FF_files, grp_c, grp_d, grp_neu, functional_grp, redox_mol, read_redox_xml, redox_states = [-2, -1, 0]):
    #def __init__(self, read_pdb, temperature, ResidueConnectivityFiles, FF_files, FF_Efield_files):
        strdir = '../'
        pdb = PDBFile( strdir + read_pdb)
        #self.index = index
        self.temperature = temperature
        self.cutoff = 1.4*nanometer

        integ_md = DrudeLangevinIntegrator(self.temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
        integ_md.setMaxDrudeDistance(0.02)  # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
        integ_junk = DrudeLangevinIntegrator(self.temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)  # this integrator won't be used, its just for Efield electric field calculation
        for bond in ResidueConnectivityFiles:
            pdb.topology.loadBondDefinitions(bond) 
            #print(bond)
        pdb.topology.createStandardBonds()

        modeller = Modeller(pdb.topology, pdb.positions)
        forcefield = ForceField(*FF_files)
        modeller.addExtraParticles(forcefield)

#        modeller2 = Modeller(pdb.topology, pdb.positions)
#        forcefield_Efield = ForceField(*FF_Efield_files)
#        modeller2.addExtraParticles(forcefield_Efield)
        
        #self.system = forcefield.createSystem(modeller.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=True)
        self.system = forcefield.createSystem(modeller.topology, nonbondedCutoff=self.cutoff, constraints=HBonds, rigidWater=True)
        self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        self.drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
        print('nbMethod : ', self.customNonbondedForce.getNonbondedMethod())

        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            type(f)
            f.setForceGroup(i)
            # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
            # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
            if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
                f.setUsesPeriodicBoundaryConditions(True)
            f.usesPeriodicBoundaryConditions()

#        # set up system2 for Efield calculation
#        self.system_Efield = forcefield_Efield.createSystem(modeller2.topology, nonbondedCutoff=self.cutoff, constraints=None, rigidWater=True) 
#        self.nbondedForce_Efield = [f for f in [self.system_Efield.getForce(i) for i in range(self.system_Efield.getNumForces())] if type(f) == NonbondedForce][0]
#        self.nbondedForce_Efield.setNonbondedMethod(NonbondedForce.PME)
#
#        for i in range(self.system_Efield.getNumForces()): 
#            f = self.system_Efield.getForce(i)
#            type(f)
#            f.setForceGroup(i)

        totmass = 0.*dalton
        for i in range(self.system.getNumParticles()):
            totmass += self.system.getParticleMass(i)

        #platform = Platform.getPlatformByName('CUDA')
        #idx=self.index.split(",")
        #properties = {'DeviceIndex': idx[0], 'Precision': 'mixed'}
        #self.simmd = Simulation(modeller.topology, self.system, integ_md, platform, properties)
        platform = Platform.getPlatformByName(platform_name)
        self.simmd = Simulation(modeller.topology, self.system, integ_md, platform)
        self.simmd.context.setPositions(modeller.positions)

        platform = self.simmd.context.getPlatform()
        platformname = platform.getName();
        print(platformname)

        # get atomlist of all electrolytes 
        #pdbresidues = [ res.name for res in pdb.topology.residues() if "grp" not in res.name and functional_grp not in res.name and redox_mol not in res.name]
        pdbresidues = [ res.name for res in pdb.topology.residues() if "grp" not in res.name and functional_grp not in res.name ]
        #pdbtest = set(pdbresidues)
        reslist = []
        for res_i in pdbresidues:
            if res_i not in reslist:
                reslist.append(res_i)
        pdbresidues_new = reslist
        print("electrolyte_list:", pdbresidues_new)
        self.solvent_list = []
        if len(pdbresidues_new) == 0:
            pass
        elif 0 < len(pdbresidues_new) :
            res_arr = ["res"+str(i) for i in range(len(pdbresidues_new))]
            for res_i in range(len(res_arr)):
                res_name = pdbresidues_new[res_i]
                self.atomlist = []
                #for res in pdb.topology.residues():
                for res in self.simmd.topology.residues():
                    if res.name == res_name:
                        for atom in res._atoms:
                            #(q_i, sig, eps) = self.nbondedForce.getParticleParameters(int(atom.index))
                            self.atomlist.append( int(atom.index) )
                self.solvent_list.extend(deepcopy(self.atomlist))

        #print(self.solvent_list, len(self.solvent_list))
        # get residue names of electrodes from .pdb
        #grp_residues = [ res.name for res in pdb.topology.residues() if "grp" in res.name]
        grp_residues = [ res.name for res in pdb.topology.residues() if (res.name == grp_neu or res.name == functional_grp)]
        print("electrode_list:", grp_residues)
        redox_residues = [ res.name for res in pdb.topology.residues() if (res.name == redox_mol)]
        if len(redox_residues) == 0:
            pass
            #redox_atomindices_cathode = []
            #redox_atomindices_anode = []
            #self.electrode = Electrode_types()
            #self.electrode.atomidx(pdb, self.nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp)
        elif 0 < len(redox_residues) :
            redox_atomtypes_list = list(dict.fromkeys([ atom.name for res in pdb.topology.residues() if (res.name == redox_mol) for atom in res._atoms ]))
            redox_chainID_list = list(dict.fromkeys([ res.chain.index for res in pdb.topology.residues() if (res.name == redox_mol) ]))
            self.Nredox_cathode = len([ res.name for res in pdb.topology.residues() if (res.name == redox_mol and res.chain.index == redox_chainID_list[0])])
            print("redox_list:", redox_residues, len(redox_residues))
            redoxstate_charges_dictionary = get_charges_of_redox_states(read_redox_xml, redox_states, redox_atomtypes_list)
            #self.redox_molecules_atomindices = get_atomindex_redox_molecules(pdb, redox_mol)
            self.redox_molecules_atomindices = get_atomindex_redox_molecules(self.simmd.topology.residues(), redox_mol)
            redox_atomindices_cathode = self.redox_molecules_atomindices[:self.Nredox_cathode] 
            redox_atomindices_anode = self.redox_molecules_atomindices[self.Nredox_cathode:]
            print(redox_atomindices_cathode, len(redox_atomindices_cathode))
            print(redox_atomindices_anode, len(redox_atomindices_anode))
            self.redox_molecules_atomcharges, self.redox_molecules_redoxstates = get_charges_redoxstates_from_atomindex(self.redox_molecules_atomindices, self.nbondedForce)
            self.redox_molecules = []
            for i_redox in range(len(redox_residues)):
                self.redox_mol_i = Redox_class(redoxstate_charges_dictionary, self.redox_molecules_atomindices[i_redox], self.redox_molecules_redoxstates[i_redox])
                self.redox_molecules.append(self.redox_mol_i)

            #self.electrode = Electrode_types(redox_atomindices_cathode, redox_atomindices_anode)
            #self.electrode.atomidx( pdb, self.nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp)

        # get atom indices of all electrodes
        #print(redox_atomindices_cathode, redox_atomindices_anode)
        #self.electrode = Electrode_types(redox_atomindices_cathode, redox_atomindices_anode)
        #self.electrode.atomidx( pdb, self.nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp)
        self.electrode = Electrode_types()
        self.electrode.atomidx(pdb, self.nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp)
        self.graph = deepcopy(self.electrode.grpc_cathode)
        self.graph.extend(deepcopy(self.electrode.dummy[:int(len(self.electrode.dummy)/2)]))
        self.graph.extend(deepcopy(self.electrode.grpc_anode))
        self.graph.extend(deepcopy(self.electrode.dummy[int(len(self.electrode.dummy)/2): len(self.electrode.dummy)]))
        self.grpc = deepcopy(self.electrode.grpc_cathode)
        self.grpc.extend(deepcopy(self.electrode.grpc_anode))
        print(len(self.graph))
        #print("sheets",len(self.grpc), len(self.graph), len(self.grph), len(self.grp_dummy))

	# write initial pdb with drude oscillators
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)	
        #position = state.getPositions()
        self.initialPositions = state.getPositions()
        self.simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        PDBFile.writeFile(self.simmd.topology, self.initialPositions, open('start_drudes'+ str(int(self.temperature)) +'.pdb', 'w'))
        #PDBFile.writeFile(self.simmd.topology, position, open('start_drudes.pdb', 'w'))
        
        self.simmd.reporters = []
        self.simmd.reporters.append(DCDReporter('md_nvt' + str(int(self.temperature)) + '.dcd', int(ntimestep_write)))
        self.simmd.reporters.append(CheckpointReporter('md_nvt' + str(int(self.temperature)) + '.chk', 10000))
        self.simmd.reporters[1].report(self.simmd,state)

        self.flagexclusions = {}

    def initialize_energy(self):
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
        print('Initial Energy')
        print(str(state.getKineticEnergy()))
        print(str(state.getPotentialEnergy()))
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    def equilibration(self):
	# print('Equilibrating...')
	# for i in range(5000):
	#     self.simmd.step(1000)
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
        position = state.getPositions()
        self.simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        # PDBFile.writeFile(self.simmd.topology, position, open(strdir+'equil_nvt.pdb', 'w'))
        state = self.simmd.context.getState(getPositions=True)
        initialPositions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(initialPositions)

        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True,getParameters=True)
        print('Equilibrated Energy')
        print(str(state.getKineticEnergy()))
        print(str(state.getPotentialEnergy()))
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

    def exlusionNonbondedForce(self, graph_cat, graph_an):
    #******* JGM ****************
    # add exclusions for intra-sheet non-bonded interactions.

    # first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
    # allow the same exclusion to be added twice
        for i in range(self.customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = self.customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            self.flagexclusions[string1]=1
            self.flagexclusions[string2]=1

    # now add exclusions for every atom pair in each sheet if we don't already have them
    #cathode first.
        for list_i in range(int(len(graph_cat))):
            for list_j in range(int(len(graph_cat))):
                #if graph[list_j] == graph[list_i]:
                #print(len(graph_cat[list_i]), len(graph_cat[list_j]))
                if list_j == list_i:
                    graph_temp = graph_cat[list_i]
                    for i in range(len(graph_temp)):
                        indexi = graph_temp[i]
                        for j in range(i+1,int(len(graph_temp))):
                            indexj = graph_temp[j]
                            string1=str(indexi)+"_"+str(indexj)
                            string2=str(indexj)+"_"+str(indexi)
                            if string1 in self.flagexclusions and string2 in self.flagexclusions:
                                continue
                            else:
                                self.customNonbondedForce.addExclusion(indexi,indexj)
                                self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                        #self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)
                elif list_j > list_i:
                    graph_temp = graph_cat[list_i]
                    grph = graph_cat[list_j]
                    for i in range(len(graph_temp)):
                        indexi = graph_temp[i]
                        #for j in range(i+1,int(len(grph)/2)):
                        for j in range(len(grph)):
                            indexj = grph[j]
                            string1=str(indexi)+"_"+str(indexj)
                            string2=str(indexj)+"_"+str(indexi)
#                            if string1 in self.flagexclusions and string2 in self.flagexclusions:
#                                continue
#                            else:
#                                self.customNonbondedForce.addExclusion(indexi,indexj)
#                                self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                            self.customNonbondedForce.addExclusion(indexi,indexj)
                            self.nbondedForce.addException(indexi,indexj,0,1,0,True)

# now anode
        for list_i in range(int(len(graph_an))):
            for list_j in range(int(len(graph_an))):
                #print(len(graph_an[list_i]), len(graph_an[list_j]))
                if list_j == list_i:
                    graph_temp = graph_an[list_i]
                    for i in range(len(graph_temp)):
                        indexi = graph_temp[i]
                        for j in range(i+1,int(len(graph_temp))):
                            indexj = graph_temp[j]
                            string1=str(indexi)+"_"+str(indexj)
                            string2=str(indexj)+"_"+str(indexi)
                            if string1 in self.flagexclusions and string2 in self.flagexclusions:
                                continue
                            else:
                                self.customNonbondedForce.addExclusion(indexi,indexj)
                                self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                elif list_j > list_i:
                    graph_temp = graph_an[list_i]
                    grph = graph_an[list_j]
                    for i in range(len(graph_temp)):
                        indexi = graph_temp[i]
                        for j in range(len(grph)):
                            indexj = grph[j]
                            string1=str(indexi)+"_"+str(indexj)
                            string2=str(indexj)+"_"+str(indexi)
                            self.customNonbondedForce.addExclusion(indexi,indexj)
                            self.nbondedForce.addException(indexi,indexj,0,1,0,True)                            
#                            if string1 in self.flagexclusions and string2 in self.flagexclusions:
#                                continue
#                            else:
#                                self.customNonbondedForce.addExclusion(indexi,indexj)
#                                self.nbondedForce.addException(indexi,indexj,0,1,0,True)

    def exlusionNonbondedForce1(self, graph):
    #******* JGM ****************
    # add exclusions for intra-sheet non-bonded interactions.

    # first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
    # allow the same exclusion to be added twice
        for i in range(self.customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = self.customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            self.flagexclusions[string1]=1
            self.flagexclusions[string2]=1

    # now add exclusions for every atom pair in each sheet if we don't already have them
    #cathode first.
        for i in range(int(len(graph)/2)):
            indexi = graph[i]
            for j in range(i+1,int(len(graph)/2)):
                indexj = graph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in self.flagexclusions and string2 in self.flagexclusions:
                    continue
                else:
                    self.customNonbondedForce.addExclusion(indexi,indexj)
                    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                    #self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)
    #now anode
        for i in range(int(len(graph)/2),len(graph)):
            indexi = graph[i]
            for j in range(i+1,int(len(graph)/2)):
                indexj = graph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in self.flagexclusions and string2 in self.flagexclusions:
                    continue
                else:
                    self.customNonbondedForce.addExclusion(indexi,indexj)
                    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                    #self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)

    def exlusionNonbondedForce2(self, graph, grph):
    #******* JGM ****************
    # add exclusions for non-bonded interactions between sheets.
    
    # first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
    # allow the same exclusion to be added twice
        for i in range(self.customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = self.customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            self.flagexclusions[string1]=1
            self.flagexclusions[string2]=1

    # now add exclusions for every atom pair in each sheet if we don't already have them
    #cathode first.
        for i in range(int(len(graph)/2)):
            indexi = graph[i]
            #for j in range(i+1,int(len(grph)/2)):
            for j in range(int(len(grph)/2)):
                indexj = grph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                self.customNonbondedForce.addExclusion(indexi,indexj)
                self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                #if string1 in self.flagexclusions and string2 in self.flagexclusions:
                #    continue
                #else:
                #    self.customNonbondedForce.addExclusion(indexi,indexj)
                #    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                    #self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)
    #now anode
        for i in range(int(len(graph)/2),len(graph)):
            indexi = graph[i]
            #for j in range(i+1,int(len(graph)/2)):
            for j in range(int(len(grph)/2),len(grph)):
                indexj = grph[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                self.customNonbondedForce.addExclusion(indexi,indexj)
                self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                #if string1 in self.flagexclusions and string2 in self.flagexclusions:
                #    continue
                #else:
                #    self.customNonbondedForce.addExclusion(indexi,indexj)
                #    self.nbondedForce.addException(indexi,indexj,0,1,0,True)
                #    #self.nbondedForce_Efield.addException(indexi,indexj,0,1,0,True)    

    def initializeCharge(self, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, small, cell_dist):
        sum_Qi_cat = 0.
        sum_Qi_an = 0.
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            (q_i, sig, eps) = self.nbondedForce.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = 1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv + small
                sum_Qi_cat += q_i
            else:  # anode
                q_i = -1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + + Voltage/cell_dist ) * conv - small
                sum_Qi_an += q_i
            self.nbondedForce.setParticleParameters(index, q_i, 1.0 , 0.0)
 
        #self.nbondedForce.updateParametersInContext(self.simEfield.context)
        self.nbondedForce.updateParametersInContext(self.simmd.context)
        return sum_Qi_cat, sum_Qi_an

    #def ConvergedCharge(self, Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, tol):
    def ConvergedCharge(self, Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol):
        rms = 0.0
        flag_conv = -1
        self.ana_Q_Cat, self.ana_Q_An = get_induced_q_analytical(z_L, z_R, cell_dist, self.nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, self.solvent_list)
        #ind_Q = get_Efield(self.solvent_list)
        #self.ana_Q_Cat, self.ana_Q_An = ind_Q.induced_q( z_L, z_R, cell_dist, self.nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv)
        print('Analytical Q_Cat, Q_An :', self.ana_Q_Cat, self.ana_Q_An)

        for i_step in range(Niter_max):
            print("charge iteration", i_step)
 
            state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
            for j in range(self.system.getNumForces()):
                f = self.system.getForce(j)
                print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
#            state2 = self.simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
#            for j in range(self.system_Efield.getNumForces()):
#                    f = self.system_Efield.getForce(j)
#                    print(type(f), str(self.simEfield.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
# 
#            forces = state2.getForces()
            forces = state.getForces()
            nbondedForce_PE_old = self.simmd.context.getState(getEnergy=True, groups=2**5).getPotentialEnergy()
            #positions= state.getPositions()
            for i_atom in range(Ngraphene_atoms):
                    index = graph[i_atom]
                    #(q_i_old, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
                    (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(index)
                    q_i_old = q_i_old
                    E_z = ( forces[index][2]._value / q_i_old._value ) if q_i_old._value != 0 else 0
                    #E_z = ( forces[index][2]._value / q_i_old._value ) if abs(q_i_old._value) > 1e-5 else 0.
                    E_i_external = E_z
                    #print(i_atom, area_atom, E_i_external, Lgap, forces[index])
 
                    # when we switch to atomic units on the right, sigma/2*epsilon0 becomes 4*pi*sigma/2 , since 4*pi*epsilon0=1 in a.u.
                    if i_atom < Ngraphene_atoms / 2:
                        q_i = 2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv
                    else:  # anode
                        q_i = -2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv
 
                    # Make sure calculated charge isn't crazy
                    if abs(q_i) > q_max:
                        # this shouldn't happen.  If this code is run, we might have a problem
                        # for now, just don't use this new charge
                        q_i = q_i_old._value
                        print("ERROR: q_i > q_max: {:f} > {:f}".format(q_i, q_max))
 
                    #self.nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
                    self.nbondedForce.setParticleParameters(index, q_i, sig, eps)
                    #rms += (q_i - q_i_old._value)**2
 
            #self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
            self.nbondedForce.updateParametersInContext(self.simmd.context)

            self.sumq_cathode, self.sumq_anode = self.FinalCharge(Ngraphene_atoms, graph, args, i, chargesFile)
            print( 'total charge on graphene (cathode,anode):', self.sumq_cathode, self.sumq_anode )
            self.Scale_charge( Ngraphene_atoms, graph, self.ana_Q_Cat, self.ana_Q_An, self.sumq_cathode, self.sumq_anode )
            #state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
            nbondedForce_PE_new = self.simmd.context.getState(getEnergy=True, groups=2**5).getPotentialEnergy()
            print('electrostatic E:', nbondedForce_PE_old, nbondedForce_PE_new)
            rms = abs( nbondedForce_PE_new._value - nbondedForce_PE_old._value)
            print('rms', rms)
            #rms += (q_i - q_i_old._value)**2
 
            #rms = (rms/Ngraphene_atoms)**0.5
            if rms < tol:
                flag_conv = i_step
                break
    # warn if not converged
        if flag_conv == -1:
            print("Warning:  Electrode charges did not converge!! rms: %f" % (rms))
        else:
            print("Steps to converge: " + str(flag_conv + 1))


#    def ConvergedCharge(self, Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, tol):
#        rms = 0.0
#        flag_conv = -1
#        for i_step in range(Niter_max):
#            print("charge iteration", i_step)
#
#            state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
#            for j in range(self.system.getNumForces()):
#                f = self.system.getForce(j)
#                print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
##            state2 = self.simEfield.context.getState(getEnergy=True,getForces=True,getPositions=True)
##            for j in range(self.system_Efield.getNumForces()):
##                    f = self.system_Efield.getForce(j)
##                    print(type(f), str(self.simEfield.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
##
##            forces = state2.getForces()
#            forces = state.getForces()
#            #positions= state.getPositions()
#            for i_atom in range(Ngraphene_atoms):
#                    index = graph[i_atom]
#                    #(q_i_old, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
#                    (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(index)
#                    q_i_old = q_i_old
#                    E_z = ( forces[index][2]._value / q_i_old._value ) if q_i_old._value != 0 else 0
#                    #E_z = ( forces[index][2]._value / q_i_old._value ) if abs(q_i_old._value) > 1e-5 else 0.
#                    E_i_external = E_z
#                    #print(i_atom, area_atom, E_i_external, Lgap, forces[index])
#
#                    # when we switch to atomic units on the right, sigma/2*epsilon0 becomes 4*pi*sigma/2 , since 4*pi*epsilon0=1 in a.u.
#                    if i_atom < Ngraphene_atoms / 2:
#                        q_i = 2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv
#                    else:  # anode
#                        q_i = -2.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + E_i_external) * conv
#
#                    # Make sure calculated charge isn't crazy
#                    if abs(q_i) > q_max:
#                        # this shouldn't happen.  If this code is run, we might have a problem
#                        # for now, just don't use this new charge
#                        q_i = q_i_old._value
#                        print("ERROR: q_i > q_max: {:f} > {:f}".format(q_i, q_max))
#
#                    #self.nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
#                    self.nbondedForce.setParticleParameters(index, q_i, sig, eps)
#                    rms += (q_i - q_i_old._value)**2
#
#            #self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
#            self.nbondedForce.updateParametersInContext(self.simmd.context)
#
#    # warn if not converged
#        if flag_conv == -1:
#            print("Warning:  Electrode charges did not converge!! rms: %f" % (rms))
#        else:
#            print("Steps to converge: " + str(flag_conv + 1))

    def FinalCharge(self, Ngraphene_atoms, graph, args, i, chargesFile):
        sumq_cathode=0
        sumq_anode=0
        print('Final charges on graphene atoms')
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            #(q_i, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
            (q_i, sig, eps) = self.nbondedForce.getParticleParameters(index)
            self.nbondedForce.setParticleParameters(index, q_i, 1.0 , 0.0)
 
            if i_atom < Ngraphene_atoms / 2:
                # print charge on one representative atom for debugging fluctuations
                #if i_atom == 100:
                #    print('index, charge, sum',index, q_i , sumq_cathode )
                sumq_cathode += q_i._value
            else:
                # print charge on one representative atom for debugging fluctuations
                #if i_atom == Ngraphene_atoms/2 + 100:
                #    print('index, charge, sum',index, q_i, sumq_anode )
                sumq_anode += q_i._value
 
#            # if we are on a 1000 step interval, write charges to file
#            # i starts at 0, so at i = 9, 1000 frames will have occured
#            #if i % 10 == 0:
#            if ( int(args.nstep) == 1 and i % 10 == 0 ):
#                chargesFile.write("{:f} ".format(q_i._value))
# 
#            if ( int(args.nstep) == 10 and i % 1 == 0 ):
#                chargesFile.write("{:f} ".format(q_i._value))
# 
#            if ( int(args.nstep) == 50 and i % 1 == 0 ):
#                chargesFile.write("{:f} ".format(q_i._value))
#
#        # write newline to charge file after charge write
#        #if i % 10 == 0:
#        if ( int(args.nstep) == 1 and i % 10 == 0 ):
#            chargesFile.write("\n")
#        elif ( int(args.nstep) == 10 and i % 1 == 0 ):
#            chargesFile.write("\n")
#        elif ( int(args.nstep) == 50 and i % 1 == 0 ):
#            chargesFile.write("\n")
        
        return sumq_cathode, sumq_anode
   
    def Scale_charge(self, Ngraphene_atoms, graph, ana_Q_Cat, ana_Q_An, sumq_cathode, sumq_anode):
    #def Scale_charge(self, Ngraphene_atoms, graph, Q_Cat_ind, Q_An_ind, sumq_cathode, sumq_anode, sum_Qi_cat, sum_Qi_an):
        Q_cat_scale = 0.
        Q_an_scale = 0.
        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            #(q_i_num, sig, eps) = self.nbondedForce_Efield.getParticleParameters(index)
            (q_i_num, sig, eps) = self.nbondedForce.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = q_i_num * (ana_Q_Cat/ sumq_cathode) if abs(sumq_cathode) != 0 else q_i_num *0.
                #q_i = q_i_num * ((Q_Cat_ind + sum_Qi_cat)/ sumq_cathode)
                Q_cat_scale += q_i._value
            else:  # anode
                q_i = q_i_num * (ana_Q_An/ sumq_anode) if abs(sumq_anode) != 0 else q_i_num *0.
                #q_i = q_i_num * ((Q_An_ind + sum_Qi_an)/ sumq_anode)
                Q_an_scale += q_i._value
            #self.nbondedForce_Efield.setParticleParameters(index, q_i, sig, eps)
            self.nbondedForce.setParticleParameters(index, q_i, sig, eps)
        #self.nbondedForce_Efield.updateParametersInContext(self.simEfield.context)
        self.nbondedForce.updateParametersInContext(self.simmd.context)
        print( 'Updated charge on cathode, anode:', Q_cat_scale, Q_an_scale )
        return Q_cat_scale, Q_an_scale


    def PrintFinalEnergies(self):
        self.nbondedForce.updateParametersInContext(self.simmd.context)
 
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
        for j in range(self.system.getNumForces()):
            f = self.system.getForce(j)
            print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))


    def Charge_solver(self, Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol):
        print("graph", len(graph))
        self.ConvergedCharge( Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, tol )
        self.sumq_cathode, self.sumq_anode = self.FinalCharge(Ngraphene_atoms, self.graph, args, i, chargesFile)
        print( 'total charge on graphene (cathode,anode):', self.sumq_cathode, self.sumq_anode )
        #print('Charges converged, Energies from full Force Field')
        #self.PrintFinalEnergies()
        self.ana_Q_Cat, self.ana_Q_An = get_induced_q_analytical(z_L, z_R, cell_dist, self.nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, self.solvent_list)
        #ind_Q = get_Efield(self.solvent_list)
        #self.ana_Q_Cat, self.ana_Q_An = ind_Q.induced_q( z_L, z_R, cell_dist, self.nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv)

        print('Analytical Q_Cat, Q_An :', self.ana_Q_Cat, self.ana_Q_An)
        self.Scale_charge( Ngraphene_atoms, graph, self.ana_Q_Cat, self.ana_Q_An, self.sumq_cathode, self.sumq_anode )

    def MonteCarlo_redox(self, ntrials, naccept, Ngraphene_atoms, Niter_max, graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol, redox_mol_i, redox_states, ET_i, PE_old, E_fermi, EA_neu, EA_red1):

        print("Starting", ET_i, "at", self.redox_molecules[redox_mol_i].redox_electrode)
        if self.redox_molecules[redox_mol_i].redox_electrode == "cathode":
            E_electrode = E_fermi - Voltage/2
            #E_electrode = E_fermi + Voltage/2
        if self.redox_molecules[redox_mol_i].redox_electrode == "anode":
            E_electrode = E_fermi + Voltage/2
            #E_electrode = -E_fermi - Voltage/2
        # save charges on graphene before redox
        self.backup_charges_on_graph = []
        for i_grp_atom in range(Ngraphene_atoms):
            index = graph[i_grp_atom]
            (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(index)
            self.backup_charges_on_graph.append(q_i_old._value)

        redox_state_temp = 0.
        for i_atom in range(len(self.redox_molecules[redox_mol_i].atomindex)):
            index = self.redox_molecules[redox_mol_i].atomindex[i_atom]
            (q_state_i, sig, eps) = self.nbondedForce.getParticleParameters( int(index))
            redox_state_temp += q_state_i._value
        redox_state_temp = round(redox_state_temp)
        print("temperary redox state", redox_state_temp)

        ntrials += 1
        self.nbondedForce.updateParametersInContext(self.simmd.context)
        self.ConvergedCharge( Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol )
        sumq_cathode_current, sumq_anode_current = self.FinalCharge(Ngraphene_atoms, graph, args, i, chargesFile)
        print( 'total charge on graphene (cathode,anode):', sumq_cathode_current, sumq_anode_current )
        #self.Charge_solver( Niter_max, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, q_max, args, i, chargesFile, z_L, z_R, cell_dist, positions, tol )

        state_new_redox = self.simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
        PE_new = state_new_redox.getPotentialEnergy()
        print("PE before and after MC move ", PE_old, PE_new._value)

        # it is important to substract out the energy differece of gas phase reduced and oxidized molecules.
        # in this case, we have excluded all intramolecular nonbonded interactions and used the same bond, angle,dihedral force constant for both redox states. so this term is zero, but in general we need to subtract these terms.
        delta_E_intra = 0
        if ET_i == 'oxidation':
            # EA_neu = Mol(0) - Mol(-1)
            # EA_red1 = Mol(-1) - Mol(-2)
            if redox_state_temp == 0:
                w = (EA_neu) + E_electrode  + (PE_new._value - PE_old - delta_E_intra)
            if redox_state_temp == -1:
                w = (EA_red1) + E_electrode  + (PE_new._value - PE_old - delta_E_intra)
        if ET_i == 'reduction':
            if redox_state_temp == -1:
                w = (-EA_neu) - E_electrode  + (PE_new._value - PE_old - delta_E_intra)
            if redox_state_temp == -2:
                w = (-EA_red1) - E_electrode  + (PE_new._value - PE_old - delta_E_intra)

        print("dE, exp(-dE/RT) ", w, numpy.exp(-(w)))
        if ( w < 0. or random.uniform(0.0,1.0) < numpy.exp(-(w)) ):
            naccept += 1

        else:
            # switch graphene charges back to previous state
            for i_grp_atom in range(Ngraphene_atoms):
                index = graph[i_grp_atom]
                (q_grp_new, sig, eps) = self.nbondedForce.getParticleParameters(index)
                q_grp_bak = self.backup_charges_on_graph[i_grp_atom]
                self.nbondedForce.setParticleParameters(index, q_grp_bak, sig, eps)
            self.nbondedForce.updateParametersInContext(self.simmd.context)

            # go back to previous redox state 
            if ET_i == 'oxidation':
                print(self.redox_molecules_redoxstates[redox_mol_i], round(redox_state_temp), "reduce back")
                self.redox_molecules[redox_mol_i].reduction(self.nbondedForce, redox_states, round(redox_state_temp), redox_mol_i, self.Nredox_cathode)
                self.nbondedForce.updateParametersInContext(self.simmd.context)
            if ET_i == 'reduction':
                print(self.redox_molecules_redoxstates[redox_mol_i], round(redox_state_temp), "oxidize back")
                self.redox_molecules[redox_mol_i].oxidation(self.nbondedForce, redox_states, round(redox_state_temp), redox_mol_i, self.Nredox_cathode)
                self.nbondedForce.updateParametersInContext(self.simmd.context)

        print("Accept ratio for redox between states 1 and 2: ", float(naccept)/float(ntrials))
        if (naccept < 0.25*ntrials) :
            ntrials = 0
            naccept = 0
        elif naccept > 0.75*ntrials :
            ntrials = 0
            naccept = 0


        #redox_state_update = 0.
        #for i_atom in range(len(self.redox_molecules[redox_mol_i].atomindex)):
        #    index = self.redox_molecules[redox_mol_i].atomindex[i_atom]
        #    (q_final, sig, eps) = self.nbondedForce.getParticleParameters( int(index))
        #    redox_state_update += q_final._value
        #print(self.redox_molecules_redoxstates[redox_mol_i], round(redox_state_temp), round(redox_state_update))



def find_charges_from_atomtype(read_redox_xml):
    #self.read_redox_xml = read_redox_xml
    chargeclass_list = []
    atomclass_list = []
    atomcharge_list = []
    atomtype_list = []
    if len(read_redox_xml) == 0:
        pass
    elif 0 < len(read_redox_xml):
        infile = open(read_redox_xml,"r")
        dummy_lines = infile.readlines()
        for i in range(len(dummy_lines)):
            line_i = dummy_lines[i].split()
            for j in line_i:
                if "charge" in j:
                    line_i_charge = line_i[1].split()
                    class_j = line_i_charge[0].split('"')
                    charge_j = j.split('"')
                    chargeclass_list.append(class_j[1])
                    atomcharge_list.append(float(charge_j[1]))
                if "element" in j:
                    line_i_element = line_i[1].split()
                    line_i_class = line_i[2].split()
                    element_j_old = line_i_element[0].split('-')
                    class_j_element = line_i_class[0].split('"')
                    element_j = element_j_old[1].replace('"','')
                    atomclass_list.append(class_j_element[1])
                    atomtype_list.append(element_j)
        atomclass_charges = dict(zip(chargeclass_list, atomcharge_list))
        atomtype_classes = dict(zip(atomtype_list, atomclass_list))
        infile.close()
    return atomtype_classes, atomclass_charges


def get_charges_of_redox_states(read_redox_xml, redox_states, redox_atomtypes):
    dictionary_atomclass_charges = {}
    dictionary_atomtype_classes = {}
    dictionary_redoxstates_charges = {}
    for i_state in range(len(redox_states)):
        atomtype_classes, atomclass_charges = find_charges_from_atomtype(read_redox_xml[i_state])
        dictionary_atomtype_classes[redox_states[i_state]] = atomtype_classes
        dictionary_atomclass_charges[redox_states[i_state]] = atomclass_charges
        charges_redox_state_i = []
        for atomname_i in redox_atomtypes:
            atomclass_state_i = dictionary_atomtype_classes[redox_states[i_state]][atomname_i]
            atomcharge_state_i = dictionary_atomclass_charges[redox_states[i_state]][atomclass_state_i]
            charges_redox_state_i.append(atomcharge_state_i)
        dictionary_redoxstates_charges[redox_states[i_state]] = charges_redox_state_i
    return dictionary_redoxstates_charges

#def get_atomindex_redox_molecules(pdb, redox_mol):
def get_atomindex_redox_molecules(simmd_topology, redox_mol):
    redox_molecules_atomindices = []
    #chain_idx = -1
    #for res in pdb.topology.residues():
    for res in simmd_topology:
        if (res.name == redox_mol):
            redox_atomidx = []
            for atom in res._atoms:
                redox_atomidx.append(atom.index)
            redox_molecules_atomindices.append(redox_atomidx)
            #if chain_idx == -1 or chain_idx == res.chain.index:
            #    chain_idx = res.chain.index
            #    print(res.name, res.id, res.chain.index, "redox_electrode_1")
            #elif res.chain.index != chain_idx:
            #    chain_idx = res.chain.index
            #    print(res.name, res.id, res.chain.index, "redox_electrode_2")
    return redox_molecules_atomindices

def get_charges_redoxstates_from_atomindex(redox_atomindex, nbondedForce):
    redox_molecules_atomcharges = []
    redox_molecules_redoxstates = []
    for res in range(len(redox_atomindex)):
        redox_idx = redox_atomindex[res]
        redox_atomidx = []
        redox_charge = []
        for atom_idx in redox_idx:
            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom_idx))
            redox_charge.append(q_i._value)
        redox_molecules_atomcharges.append(redox_charge)
        redox_molecules_redoxstates.append(round(sum(redox_charge)))
    return redox_molecules_atomcharges, redox_molecules_redoxstates

class Redox_class:
    def __init__(self, redoxstate_charges_dictionary, i_redox_molecule_atomindices, i_redox_molecule_state):
        self.redoxstate_charges_dictionary = redoxstate_charges_dictionary
        self.atomindex = i_redox_molecule_atomindices
        self.redox_state = i_redox_molecule_state

    def oxidation(self, nbondedForce, redox_states, i_redox_molecule_state, redox_mol_i, Nredox_cathode):
        if redox_mol_i < Nredox_cathode:
            self.redox_electrode = "cathode"
        elif redox_mol_i >= Nredox_cathode:
            self.redox_electrode = "anode"

        if i_redox_molecule_state < redox_states[-1]:
            self.flag = 1
            for i_atom in range(len(self.atomindex)):
                index = self.atomindex[i_atom]
                (q_state_i, sig, eps) = nbondedForce.getParticleParameters( int(index))
                redox_state_f = int(i_redox_molecule_state + 1)
                q_state_ox = self.redoxstate_charges_dictionary[redox_state_f][i_atom]
                #print(redox_state_f, q_state_ox)
                nbondedForce.setParticleParameters(index, q_state_ox, sig, eps)
        if i_redox_molecule_state == redox_states[-1]:
            self.flag = 0


    def reduction(self, nbondedForce, redox_states, i_redox_molecule_state, redox_mol_i, Nredox_cathode):
        if redox_mol_i < Nredox_cathode:
            self.redox_electrode = "cathode"
        elif redox_mol_i >= Nredox_cathode:
            self.redox_electrode = "anode"

        if i_redox_molecule_state > redox_states[0]:
            self.flag = 1
            for i_atom in range(len(self.atomindex)):
                index = self.atomindex[i_atom]
                (q_state_i, sig, eps) = nbondedForce.getParticleParameters( int(index))
                redox_state_f = int(i_redox_molecule_state - 1)
                q_state_red = self.redoxstate_charges_dictionary[redox_state_f][i_atom]
                #print(redox_state_f, q_state_red)
                nbondedForce.setParticleParameters(index, q_state_red, sig, eps)
        if i_redox_molecule_state == redox_states[0]:
            self.flag = 0


#class Electrode_types:
#    def __init__(self, redox_atomindices_cathode = [], redox_atomindices_anode = []):
#        self.all_cathode_atomindices = deepcopy(redox_atomindices_cathode)
#        self.all_anode_atomindices = deepcopy(redox_atomindices_anode)
#
#    def atomidx(self, simmd_topology, nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp):
#        if len(grp_residues) == 0:
#            pass
#        elif 0 < len(grp_residues) :
#            #grp_res_arr = ["res"+str(i) for i in range(len(grp_residues))]
#            grp_types = list(dict.fromkeys(grp_residues))
#            for grp_res_i in range(len(grp_types)):
#                grp_type_i = grp_types[grp_res_i]
#                self.chain_idx = -1
#                self.grp_idx = -1
#                self.res_idx = -1
#                self.c562_1 = -1
#                self.c562_2 = -1
#                self.grpc_cathode = []
#                self.grpc_anode = []
#                self.dummy = []
#                self.neutral = []
#                self.extra = []
#                for grp_res in simmd_topology:
#                    self.grp_list = []
#                    self.electrode_1_list = []
#                    self.electrode_2_list = []
#                    if (grp_res.name == grp_type_i):
#                        if self.chain_idx == -1 or self.chain_idx == grp_res.chain.index:
#                            self.chain_idx = grp_res.chain.index
#                            self.grp_idx = grp_res.index
#                            print(grp_res.index, grp_res.name, grp_res.id, "electrode_1")
#                            for atom in grp_res._atoms:
#                                (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
#                                self.electrode_1_list.append( int(atom.index) )
#                                self.grp_list.append( int(atom.index) )
#                            self.all_cathode_atomindices.append(self.electrode_1_list)
#                        elif grp_res.chain.index != self.chain_idx:
#                            print(grp_res.index, grp_res.name, grp_res.id, "electrode_2")
#                            for atom in grp_res._atoms:
#                                (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
#                                self.electrode_2_list.append( int(atom.index) )
#                                self.grp_list.append( int(atom.index) )
#                            self.all_anode_atomindices.append(self.electrode_2_list)
#
#                    if grp_res.name == grp_c:
#                        if self.res_idx == -1:
#                            self.res_idx = grp_res.index
#                            for atom in grp_res._atoms:
#                                self.grpc_cathode.append(int( atom.index))
#                                if atom.name == "C562":
#                                    self.c562_1 = atom.index
#                        elif grp_res.index != self.res_idx:
#                            for atom in grp_res._atoms:
#                                self.grpc_anode.append(int( atom.index))
#                                if atom.name == "C562":
#                                    self.c562_2 = atom.index
#                    if grp_res.name == grp_d:
#                        for atom in grp_res._atoms:
#                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
#                            self.dummy.append( int(atom.index) )
#                    if grp_res.name == grp_neu:
#                        for atom in grp_res._atoms:
#                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
#                            self.neutral.append( int(atom.index) )
#                    if (grp_res.name == str(functional_grp)) :
#                        for atom in grp_res._atoms:
#                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
#
#        return self.all_cathode_atomindices, self.all_anode_atomindices


class Electrode_types:
    def __init__(self, redox_atomindices_cathode = [], redox_atomindices_anode = []):
        self.all_cathode_atomindices = deepcopy(redox_atomindices_cathode)
        self.all_anode_atomindices = deepcopy(redox_atomindices_anode)

    def atomidx(self, pdb, nbondedForce, grp_residues, grp_c, grp_d, grp_neu, functional_grp):
        if len(grp_residues) == 0:
            pass
        elif 0 < len(grp_residues) :
            grp_types = list(dict.fromkeys(grp_residues))
            for grp_res_i in range(len(grp_types)):
                grp_type_i = grp_types[grp_res_i]
                self.chain_idx = -1
                self.grp_idx = -1
                self.res_idx = -1
                self.c562_1 = -1
                self.c562_2 = -1
                self.grpc_cathode = []
                self.grpc_anode = []
                self.dummy = []
                self.neutral = []
                self.extra = []
                for grp_res in pdb.topology.residues():
                    self.grp_list = []
                    self.electrode_1_list = []
                    self.electrode_2_list = []
                    if (grp_res.name == grp_type_i):
                        if self.chain_idx == -1 or self.chain_idx == grp_res.chain.index:
                            self.chain_idx = grp_res.chain.index
                            self.grp_idx = grp_res.index
                            print(grp_res.index, grp_res.name, grp_res.id, "electrode_1")
                            for atom in grp_res._atoms:
                                (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
                                self.electrode_1_list.append( int(atom.index) )
                                self.grp_list.append( int(atom.index) )
                            self.all_cathode_atomindices.append(self.electrode_1_list)
                            #self.grp_atomlist_arrays.append(self.grp_list)
                        elif grp_res.chain.index != self.chain_idx:
                            print(grp_res.index, grp_res.name, grp_res.id, "electrode_2")
                            for atom in grp_res._atoms:
                                (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
                                self.electrode_2_list.append( int(atom.index) )
                                self.grp_list.append( int(atom.index) )
                            self.all_anode_atomindices.append(self.electrode_2_list)
                            #self.grp_atomlist_arrays.append(self.grp_list)
                    if grp_res.name == grp_c:
                        if self.res_idx == -1:
                            self.res_idx = grp_res.index
                            for atom in grp_res._atoms:
                                self.grpc_cathode.append(int( atom.index))
                                if atom.name == "C562":
                                    self.c562_1 = atom.index
                        elif grp_res.index != self.res_idx:
                            for atom in grp_res._atoms:
                                self.grpc_anode.append(int( atom.index))
                                if atom.name == "C562":
                                    self.c562_2 = atom.index
                    if grp_res.name == grp_d:
                        for atom in grp_res._atoms:
                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
                            self.dummy.append( int(atom.index) )
                    if grp_res.name == grp_neu:
                        for atom in grp_res._atoms:
                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
                            self.neutral.append( int(atom.index) )
                    if (grp_res.name == str(functional_grp)) :
                        for atom in grp_res._atoms:
                            (q_i, sig, eps) = nbondedForce.getParticleParameters(int(atom.index))
                            self.extra.append( int(atom.index) )

        return self.all_cathode_atomindices, self.all_anode_atomindices


class solution_Hlist:
    def __init__(self, resname):
        self.resname = resname
        self.cation = []
        self.anion = []
        self.solvent = []
        self.He = []
    def cation_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'H' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.cation.append( int(atom.index) )
    def anion_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'B' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.anion.append( int(atom.index) )
    def solvent_hlist(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                    if 'H' in list(atom.name):
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.solvent.append( int(atom.index) )
    def vac_list(self, sim):
        for res in sim.simmd.topology.residues():
            if res.name == self.resname:
                for atom in res._atoms:
                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
                        self.He.append( int(atom.index) )


#class solution_allatom:
#    def __init__(self, resname):
#        self.resname = resname
#        self.atomlist = []
#    def res_list(self, sim):
#        for res in sim.simmd.topology.residues():
#            if res.name == self.resname:
#                for atom in res._atoms:
#                        (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(int(atom.index))
#                        self.atomlist.append( int(atom.index) )
#


class get_Efield:
    def __init__(self, alist):
        self.alist = alist
        self.efieldx = []
        self.efieldy = []
        self.efieldz = []
        self.position_z = []
        self.Q_Cat_ind = 0.
        self.Q_An_ind = 0.
        self.Q_Cat = 0.
        self.Q_An = 0.
    def efield(self, sim, forces):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            #(q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(H_idx)
            (q_i, sig, eps) = sim.nbondedForce.getParticleParameters(H_idx)
            E_x_i = ( forces[H_idx][0]._value / q_i._value ) if q_i._value != 0 else 0
            E_y_i = ( forces[H_idx][1]._value / q_i._value ) if q_i._value != 0 else 0
            E_z_i = ( forces[H_idx][2]._value / q_i._value ) if q_i._value != 0 else 0
            self.efieldx.append( E_x_i )
            self.efieldy.append( E_y_i )
            self.efieldz.append( E_z_i )
    def Pos_z(self, positions):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            self.position_z.append( positions[H_idx][2]._value )
    #def induced_q(self, eletrode_L, eletrode_R, cell_dist, sim, positions):
    def induced_q(self, eletrode_L, eletrode_R, cell_dist, nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv):
        for H_i in range(len(self.alist)):
            H_idx = self.alist[H_i]
            #(q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(H_idx)
            (q_mol_i, sig, eps) = nbondedForce.getParticleParameters(H_idx)
            self.position_z.append( positions[H_idx][2]._value )
            zR = eletrode_R - positions[H_idx][2]._value
            zL = positions[H_idx][2]._value - eletrode_L
            self.Q_Cat_ind += (zR / cell_dist)* (- q_mol_i._value)
            self.Q_An_ind += (zL / cell_dist)* (- q_mol_i._value)
#        return self.Q_Cat_ind, self.Q_An_ind

        for i_atom in range(Ngraphene_atoms):
            index = graph[i_atom]
            #(q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(index)
            (q_i, sig, eps) = nbondedForce.getParticleParameters(index)
            if i_atom < Ngraphene_atoms / 2:
                q_i = 1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
                self.Q_Cat += q_i
            else:  # anode
                q_i = -1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
                self.Q_An += q_i

        ana_Q_Cat =  self.Q_Cat_ind + self.Q_Cat
        ana_Q_An = self.Q_An_ind + self.Q_An
        return ana_Q_Cat, ana_Q_An

def get_induced_q_analytical(eletrode_L, eletrode_R, cell_dist, nbondedForce, positions, Ngraphene_atoms, graph, area_atom, Voltage, Lgap, conv, all_electrolytes_list):
    Q_Cat_ind = 0.
    Q_An_ind = 0.
    Q_Cat = 0.
    Q_An = 0.
    for H_i in range(len(all_electrolytes_list)):
        H_idx = all_electrolytes_list[H_i]
        #(q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(H_idx)
        (q_mol_i, sig, eps) = nbondedForce.getParticleParameters(H_idx)
        #self.position_z.append( positions[H_idx][2]._value )
        zR = eletrode_R - positions[H_idx][2]._value
        zL = positions[H_idx][2]._value - eletrode_L
        Q_Cat_ind += (zR / cell_dist)* (- q_mol_i._value)
        Q_An_ind += (zL / cell_dist)* (- q_mol_i._value)
     #return self.Q_Cat_ind, self.Q_An_ind

    for i_atom in range(Ngraphene_atoms):
        index = graph[i_atom]
        #(q_i, sig, eps) = sim.nbondedForce_Efield.getParticleParameters(index)
        (q_i, sig, eps) = nbondedForce.getParticleParameters(index)
        if i_atom < Ngraphene_atoms / 2:
            q_i = 1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
            Q_Cat += q_i
        else:  # anode
            q_i = -1.0 / ( 4.0 * 3.14159265 ) * area_atom * (Voltage / Lgap + Voltage/cell_dist) * conv
            Q_An += q_i

    ana_Q_Cat =  Q_Cat_ind + Q_Cat
    ana_Q_An = Q_An_ind + Q_An
    return ana_Q_Cat, ana_Q_An

def Distance(p1, p2, initialPositions):
    pos_c562_1 = initialPositions[p1]
    pos_c562_2 = initialPositions[p2]
    cell_dist = 0
    for i in range(3):
        d = pos_c562_1[i] / nanometer - pos_c562_2[i] / nanometer
        cell_dist += (d**2)

    cell_dist = cell_dist**(1/2)
    return cell_dist, pos_c562_1[2]/nanometer, pos_c562_2[2]/nanometer


class hist_Efield:
    def __init__(self, dz, zdim, zlist, Ezlist):
        self.dz = dz
        self.zdim = zdim
        self.bins = [i*self.dz for i in range(0, int(self.zdim/self.dz))]
        self.zlist = zlist
        self.Ezlist = Ezlist
        self.Ezcount_i = []
    def Efield(self):
        for bin_i in range(len(self.bins)-1):
            bin0 = self.bins[bin_i]
            bin1 = self.bins[bin_i+1]
            ztotal = [self.Ezlist[i] for i,x in enumerate(self.zlist) if bin0 < x <= bin1]
            avg_count = sum(ztotal)/len(ztotal) if len(ztotal) != 0 else 0
            self.Ezcount_i.append( avg_count )
        return self.Ezcount_i
    def save_hist(self, hlist, filename):
        ofile = open(filename, "w")
        for i in range(len(hlist)):
            line = str("{0:3.5f}".format(float(self.bins[i+1]))) + "  " + str("{0:5.8f}".format(float(hlist[i]))) + "\n"
            ofile.write(line)
        ofile.close()


def read_input(filename):
    forcefieldfolder = '../ffdir/'
    grp_c = ""
    grp_d = ""
    grp_neu = ""
    functional_grp = ""
    forcefield_files = []
    residueconnectivity_files = []
    redox_state_f_xml = []
    all_redox_states = []
    #all_EA_gas = []
    with open(filename, "r") as infile:
        file_contents = infile.readlines()
        for input_i in file_contents:
            if "charge_update" in input_i:
                n_update = input_i.split()[2]
            if "voltage" in input_i:
                volt = input_i.split()[2]
            if "temperature" in input_i:
                temperature = input_i.split()[2]
            if "time(ns)" in input_i:
                nsec = input_i.split()[2]
            if "ntimestep_write(fs)" in input_i:
                ntimestep_write = input_i.split()[2]
            if "platform_name" in input_i:
                platform_name = input_i.split()[2]
            if "E_fermi" in input_i:
                E_fermi = input_i.split()[2]
            if "redox_states" in input_i:
                connect = input_i.replace(',',' ').split()[2:]
                for state_i in connect:
                    all_redox_states.append(int(state_i))
            if "EA_neu" in input_i:
                EA_neu = input_i.split()[2]
            if "EA_red1" in input_i:
                EA_red1 = input_i.split()[2]
            #if "EA_gas" in input_i:
            #    connect = input_i.replace(',',' ').split()[2:]
            #    for state_i in connect:
            #        all_EA_gas.append(float(state_i))
            if "ResidueConnectivityFiles" in input_i:
                connect = input_i.replace(',',' ').split()[2:]
                for xml_i in connect:
                    connectivity_file_i = forcefieldfolder + xml_i
                    residueconnectivity_files.append(connectivity_file_i)
            if "FF_files" in input_i:
                connect = input_i.replace(',',' ').split()[2:]
                for xml_i in connect:
                    ff_file_i = forcefieldfolder + xml_i
                    forcefield_files.append(ff_file_i)
            if "FF_redox_state_f" in input_i:
                if input_i.split()[2] == 'None':
                    redox_state_f_xml = ''
                else:
                    connect = input_i.replace(',',' ').split()[2:]
                    for xml_i in connect:
                        redox_ff_file_i = forcefieldfolder + xml_i
                        redox_state_f_xml.append(redox_ff_file_i)
                    #redox_state_f_xml = forcefieldfolder +  input_i.split()[2]
            if "conducting_sheet" in input_i:
                grp_c = input_i.split()[2]
            if "conducting_dummy" in input_i:
                grp_d = input_i.split()[2]
            if "neutral_sheet" in input_i:
                grp_neu = input_i.split()[2]
            if "additional_group_on_graph" in input_i:
                functional_grp = input_i.split()[2]
            if "redox_at_graph" in input_i:
                redox_mol = input_i.split()[2]
    #print(residueconnectivity_files, *residueconnectivity_files)
    #print(n_update, volt, temperature, nsec, residueconnectivity_files, forcefield_files, grp_c, grp_d, grp_neu, functional_grp)

    return n_update, volt, temperature, nsec, ntimestep_write, platform_name, residueconnectivity_files, forcefield_files, grp_c, grp_d, grp_neu, functional_grp, redox_mol, redox_state_f_xml, all_redox_states, E_fermi, EA_neu, EA_red1


