#!/usr/bin/env python
# -*- encoding:utf-8 -*-

#Based on PyGamess script
import numpy 
from rdkit import Chem
import re
import contextlib
#import PubChemPy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import cclib 
import subprocess as sub 
from tempfile import mkdtemp
import shutil 
from random import choice
import os 
import textwrap


def randstr(n):
    """make a random string"""
    return ''.join(choice('abcdefghijklmnopqrstuvwxyz') for i in range(n))


class FireflyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Firefly(object):
    """FIREFLY WRAPPER"""

    def __init__(self, firefly_path=None, ase=False, debug=True, **options):
        self.ase = ase
        self.debug = debug 
        #self.err_lines = 10

            
        # search firefly_path
        # 1. find environ
        # 2. find path which include firefly820
        if firefly_path is None:
            firefly_path = os.environ.get('FIREFLY_HOME', None)

        if firefly_path is None:
            try:
                firefly_path = list(filter(lambda f: os.path.isfile(os.path.join(f, 'firefly820')), [d for d in os.environ['PATH'].split(':')]))[0]
               
                
                
            except IndexError:
                print("firefly_path not found")
                exit() 
        
        #  serch ffimpi script
        ffimpi = None
        try:
            ffimpi = [f for f in [os.path.join(d, 'ffimpi.sh') for d in os.environ['HOME'].split(':')] if os.path.isfile(f)][0]
        except IndexError:
            pass
        
        
        #SETTINGS:
        #FOR FUTURE: MAKE SETTINGS A NESTED DICT FROM PARSED MANUAL AND IT WILL BE THE DIFFERENT CLASS
        
        contrl = ["SCFTYP","RUNTYP","MPLEVL","CITYP","DFTTYP","DFTD","DFTNL","NUMDER",
                    "EXETYP","MAXIT","ICHARG","MULT","ECP","OLDECP","ICHECP","COORD",
                    "UNITS","NZVAR","LOCAL","D5","MOLPLT","PLTORB","AIMPAC","RPAC",
                     "FRIEND","NPRINT","NOSYM","INTTYP","FSTINT","REORDR","GENCON",
                     "NORMF","NORMP","ICUT","ITOL","LEXCUT","HPGRAD","WIDE","IREST","GEOM",
                    "NOSO"]
        basis = ["GBASIS","NGAUSS","NDFUNC","NFFUNC","NPFUNC","DIFFSP","DIFFS","ELNEG",
                   "POLAR","SPLIT2","SPLIT3","EXTFIL"]
        statpt = ["METHOD","OPTTOL","NSTEP","NOREG","IFREEZ","DXMAX","TRUPD","TRMAX",
                    "TRMIN","IFOLOW","STPT","STSTEP","HESS","IHREP","UPHESS","HSSEND",
                     "HUPTOL","NPRT","NPUN","IDUMP","IREST","RESTAR","MAXDII","NSKIP",
                     "KEEPHS","REGTOL","NODIAG","MIXED","NOGDUP","FLAGS","MSONLY","NNEG",
                    "RMIN","RMAX","RLIM","PURIFY","PROJCT","ITBMAT","CNSTOL","FMAXT",
                     "MOVIE"]
        system = ["TIMLIM","MWORDS","MEMORY","SAFMEM","MASMEM","SHMEM","WSCTL","MAXWS",
                     "DECOMM","IDLE","BLKSIZ","LDAR","NODEL","FASTF","TRUNCF","FLUSH",
                     "ASYNC","AIOBUF","AIOPTY","SPLITF","VOLSIZ","MXIOB","MEMF","FRESHF",
                     "NOSEOF","MEMCPY","IOFLGS","IORTRY","MXBCST","MPISNC","MXBNUM",
                     "LENSNC","AOINTS","MKLNP","MKLAFF","XPBIND","MKLSMP","BLAS3","NP",
                    "KDIAG","NOJAC","L2SIZE","FPECHK"]
        scf = ["DIRSCF","FDIFF","FDCTRL","XFDIFF","XFDNR","UHFNOS","MVOQ","NPUNCH",
                  "JKMAT","NCONV","ENGTHR","DIITHR","SOGTHR","DENTOL","DIIS","SOSCF",
                  "EXTRAP","DAMP","SHIFT","FSHIFT","RSTRCT","DEM","FSTDII","ETHRSH",
                  "MAXDII","DIIMOD","DIIERR","DIITOL","SOGTOL","SODIIS","DIONCE","DEMCUT",
                 "DMPCUT","VTSCAL","SCALF","MAXVT","VTCONV","NCO","NSETO","NO","NPAIR",
                  "CICOEF","COUPLE","F","ALPHA","BETA"]
        
        settings = {"contrl":contrl, "basis":basis, "statpt":statpt, "system":system, "scf":scf}
        
        #DEFAULT SETTINGS:
        self.chronics = os.path.join(os.environ["HOME"], "chro")
        self.firefly_path = firefly_path
        self.firefly = os.path.join(self.firefly_path, "firefly820")
        self.contrl = {'DFTTYP':'B3LYP', 'MAXIT':'200','ICHARG':'0','MULT':'1','RUNTYP':'GRADIENT','SCFTYP': 'RHF'}
        self.basis = {'GBASIS': 'N31', 'NGAUSS': '6', 'NDFUNC': '1'}
        self.statpt = {'NSTEP': '100', 'OPTTOL': '0.0001'}
        self.system = {'MWORDS': '30'}
        self.scf = {'DIIS' : '.T.', 'SOSCF' : '.f.', 'DIRSCF' : '.f.'}
        self.cis = {'CIS': '1'}
        self.extfil = None
        
        #ASE-NEB
        #TODO
        
        #ROUTINES
        #TODO
        for key,value in settings.items():
            self.__getattribute__(key).update({k:options.get(k, {}) for k in set(options) & set(value)})

    def parse_fireout(self, mol):
        parser = cclib.io.ccopen(self.fireout)
        data = parser.parse()
        nmol = Chem.Mol(mol)
        conf = nmol.GetConformer(0)
        self.energy = float(data.scfenergies[-1])
        if self.contrl["RUNTYP"] in ["GRADIENT","HESSIAN","OPTIMIZE","SADPOINT","IRC","GRADEXTR","DRC","RAMAN"]:
            self.forces = numpy.negative(data.grads[-1])
        nmol.SetDoubleProp("energy", float(data.scfenergies[-1]))
        n = data.atomnos.tolist()
        c = data.atomcoords[-1].tolist()
        for x, y in zip(n, c):
            conf.SetAtomPosition(x,y)
        print("Total Energy = " + str(self.energy))
        return nmol

    @contextlib.contextmanager
    def make_temp_directory(self):
        self.tempdir = mkdtemp()
        try:
            yield self.tempdir
        finally:
            if self.debug:
                print(self.tempdir)
            else:
                shutil.rmtree(self.tempdir)
            
    def exec_firefly(self, mol):
        firein = self.write_file(mol)
        p = sub.Popen("ulimit -s 4096 {0}".format(self.firefly), shell=True)
        #  exec firefly820
        if self.extfil is None:
            com = "{0} -prealloc:485 -r -f -p -p4pg {1}/procgrp -ex {1} -i {2} -o {3} -t {4}/{5}".format(self.firefly, str(self.firefly_path), str(self.firein), str(self.fireout), self.tempdir, self.jobname)
        else:
            com = "{0} -prealloc:485 -r -f -p -p4pg {1}/procgrp -ex {1} -i {2} -o {3} -t {4}/{5} -b {1}{6}.lib".format(self.firefly, str(self.firefly_path), str(self.firein), str(self.fireout), self.tempdir, self.jobname, self.extfil)
        p = sub.Popen(com.split(), shell=False).wait() 
        new_mol = self.parse_fireout(mol)
        return new_mol
        #except AttributeError:
            #sub.run("cd ~/ && bash cleanipcs", shell=True)
        #if self.ase and not self.debug:
            #os.unlink(self.firein)
            #os.unlink(self.fireout)
    def get_name(self, mol):
        name = Chem.MolToSmiles(mol)
        results = pcp.get_compounds(name, 'smiles')
        return results[0].name
            
    
    def run(self, mol):
        if self.ase:
            with self.make_temp_directory() as r:
                new_mol = self.exec_firefly(mol)
                return new_mol  
        else:
            self.tempdir = "/tmp/" 
            new_mol = self.exec_firefly(mol)
            return new_mol
        
        ###############################################
        #WHAT WE HAVE THERe IS a SET OF PROTO-RULES  #
        #FOR MANAGING DIFFERENT CALCULATION TYPES,    #
        #LATER THEY WILL BECOME THE DIFFERENT CLASSES.#
        ###############################################
        
        
        #On a ce qu'on pouvait appeller une sorte de connerie avec 
        #minisculisation des attributs de cette classe dans la définition 
        #de la fonction d'impripation d'une section introduisante des paramètres.
        #Mais on s'en fout.
        
    def print_header(self):
        """ firefly header"""
        header = textwrap.indent(textwrap.fill("{}{}{}{}".format(self.print_section('contrl'),
                                 self.print_section('basis'),
                                 self.print_section('system'),
                                 self.print_section('scf')), 60, replace_whitespace=False), prefix=" ")
        if self.contrl['RUNTYP'] == 'OPTIMIZE':
            header += self.print_section('statpt')

        if self.contrl.get('CITYPE', None) == 'CIS':
            header += self.print_section('cis')
        return header

    def print_section(self, pref):
        d = getattr(self, pref)
        section = "${} ".format(pref)
        for k, v in d.items():
            section += "{}={} ".format(k, v)
        section += "$end\n"
        return section

    def atom_section(self, mol):
        # self.contrl['icharg'] = mol.GetFormalCharge()
        conf = mol.GetConformer(0)
        section = ""
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            section += "{:<3} {:>4}.0   {:> 15.10f} {:> 15.10f} {: 15.10f} \n".format(atom.GetSymbol(), atom.GetAtomicNum(), pos.x, pos.y, pos.z)
        return section

    def input(self, mol):
        return "{0}\n $DATA\n{1}\nC1\n{2} $END\n".format(self.print_header(), self.jobname,
                                                        self.atom_section(mol))

    def write_file(self, mol):        
        if self.ase:
            self.jobname = randstr(6)
            self.firein = self.tempdir + "/{0}.inp".format(self.jobname)
            self.fireout = self.tempdir + "/{0}.out".format(self.jobname)
        else: 
            self.jobname = input("Please, enter jobname:")
            self.dir = os.path.join(self.chronics, self.jobname)
            if os.path.isdir(os.path.join(self.chronics, self.jobname)):
                shutil.rmtree(os.path.join(self.chronics, self.jobname))
            os.mkdir(self.dir)
            self.firein = self.dir + "/{0}.inp".format(self.jobname)
            self.fireout = self.chronics + "/{0}/{0}.out".format(self.jobname)
        with open(self.firein, "w+") as f:
            f.write(self.input(mol))
        return self.firein

    def set_basis(self, basis_type=None): 
        #print("""Available basis sets:""")
        if basis_type == None:
            basis_type = input("Please, make your choise of the basis set:").upper()
        else:
            basis_type = basis_type.upper()
        if basis_type in ["STO3G", "STO-3G"]:
            self.basis = {'GBASIS': 'sto', 'NGAUSS': '3'}
        elif basis_type in ["321G", "3-21G"]:
            self.basis = {'GBASIS': 'N21', 'NGAUSS': '3'}
        elif basis_type in ["631G", "6-31G"]:
            self.basis = {'GBASIS': 'N31', 'NGAUSS': '6'}
        elif basis_type in ["6311G", "6-311G"]:
            self.basis = {'GBASIS': 'N311', 'NGAUSS': '6'}
        elif basis_type in ["631G*", "6-31G*", "6-31G(D)", "631G(D)"]:
            self.basis = {'GBASIS': 'N31', 'NGAUSS': '6', 'NDFUNC': '1'}
        elif basis_type in ["631G**", "6-31G**", "631GDP", "6-31G(D,P)", "631G(D,P)"]:
            self.basis = {'GBASIS': 'N31', 'NGAUSS': '6', 'NDFUNC': '1', 'NPFUNC': '1'}
        elif basis_type in ["631+G**", "6-31+G**", "631+GDP", "6-31+G(D,P)", "631+G(D,P)"]:
            self.basis = {'GBASIS': 'n31', 'NGAUSS': '6', 'NDFUNC': '1', 'NPFUNC': '1', 'DIFFSP': '.t.', }
        elif basis_type in["AM1"]:
            self.basis = {'GBASIS': 'am1'}
        elif basis_type in ["PM3"]:
            self.basis = {'GBASIS': 'pm3'}
        elif basis_type in ["MNDO"]:
            self.basis = {'GBASIS': 'mndo'}
        elif basis_type in ["CCPVDZ", "CC-PVDZ"]:
            self.basis = {'extfil': '.t.', 'GBASIS': 'cc-pvdz'}
            self.contrl.update({"D5": ".t."})
            self.extfil = os.path.join(self.firefly_path, "/basis/{0}".format(self.basis["GBASIS"]))
        else:
            print("basis type not found")
        return self.basis

    def set_runtyp(self, runtype):      
        self.contrl['RUNTYP'] = runtype

    def set_scftyp(self, scftype):
        self.contrl['SCFTYP'] = scftype
        
    def set_charge(self, system_charge):
        self.contrl['ICHARG'] = str(system_charge)
    
    def set_mult(self, multiplicity):
        self.contrl['MULT'] = str(multiplicity)
        
if __name__ == '__main__':

    f = Firefly()
    mol = Chem.MolFromMolFile("examples/ethane.mol", removeHs=False)
    try:
        newmol = f.run(mol)
    except FireflyError as gerr:
        print(gerr.value)

    print(newmol.GetProp("total_energy"))
