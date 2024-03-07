import numpy as np
import numpy as np
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from CADETProcess.processModel import (
    ComponentSystem, Inlet, Cstr, TubularReactor, Outlet, FlowSheet, Process, TubularReactor,
    Langmuir, LumpedRateModelWithPores, StericMassAction,)

from CADETProcess.simulator import Cadet
from CADETProcess.reference import ReferenceIO
from CADETProcess.comparison import Comparator

# Variables
Q_cmh = 400
Q_mLmin = Q_cmh*0.25**2*pi/60
Q_m3s = Q_mLmin/(60*1e6)
V_col = 20*0.25**2*pi*1e-6
V_tracer = 0.6e-06
Conc_molL = 13.09/150000


def generate_poros_xs(component_system):
    binding_model = StericMassAction(component_system)
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [1, 1e4]
    binding_model.desorption_rate = [1, 1]
    binding_model.capacity = 102.54
    binding_model.steric_factor = [0, 50]
    binding_model.characteristic_charge = [1, 11.5]

    column = LumpedRateModelWithPores(component_system, name='column')
    column.binding_model = binding_model
    column.bed_porosity = 0.60 # analytisch
    column.particle_porosity = 0.65 # analtytisch
    column.particle_radius = 50*1e-06 # 50 um
    column.film_diffusion = [1e-04, 2.11618851e0, 2.11618851e0]
    column.length = 0.2
    column.diameter = 0.005
    column.axial_dispersion =  3.420580134283602e-07 # van Deemter
    column.c = [10, 0, 0]
    column.q = [binding_model.capacity, 0, 0]

    return column

def generate_sec_column(component_system):
    column = LumpedRateModelWithPores(component_system, name='column')
    column.bed_porosity = 0.60 # analytisch
    column.particle_porosity = 0.65 # analtytisch
    column.pore_accessibility = [1, 1, 0.01]
    column.particle_radius = 50*1e-06 # 50 um
    column.film_diffusion = [1e-04, 2.11618851e0, 2.11618851e0]
    column.length = 0.2
    column.diameter = 0.005
    column.axial_dispersion =  3.420580134283602e-07 # van Deemter
    column.c = [10, 0, 0]

    return column

def generate_akta(component_system, column=None):
    inlet = Inlet(component_system, name='inlet')
    inlet.flow_rate = Q_m3s
    outlet = Outlet(component_system, name='outlet')

    # add units
    flowsheet = FlowSheet(component_system)
    flowsheet.add_unit(inlet)
    if column:
        flowsheet.add_unit(column)
    flowsheet.add_unit(outlet)

    # connect units
    if column:
        flowsheet.add_connection(inlet, column)
        flowsheet.add_connection(column, outlet)
    else:
        flowsheet.add_connection(inlet, outlet)
    return flowsheet



def generate_isocratic_process(flowsheet):
    # Timepoints of different events
    CV_time = ((flowsheet.column.diameter*0.5)**2*flowsheet.column.length*pi)/Q_m3s # this defines the duration of 1 CV

    process = Process(flowsheet, name='process isocractic')
    process.cycle_time = CV_time*1.5
    process.add_event('inject', 'flow_sheet.inlet.c',
        [10, Conc_molL*1e3, Conc_molL*1e3], time=0)
    process.add_event('elution', 'flow_sheet.inlet.c', [10, 0, 0], CV_time*0.05)

    return process
