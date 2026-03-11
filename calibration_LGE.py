import numpy as np
import numpy as np
from math import pi
import pandas as pd


from CADETProcess.processModel import (
    ComponentSystem, Inlet, Cstr, TubularReactor, Outlet, FlowSheet, Process, TubularReactor, GeneralRateModel,
    Langmuir, LumpedRateModelWithPores, StericMassAction,)

from CADETProcess.simulator import Cadet
from CADETProcess.reference import ReferenceIO
from CADETProcess.comparison import Comparator

# Variables
Q_cmh = 150
Q_mLmin = Q_cmh*1**2*pi/60
Q_m3s = Q_mLmin/(60*1e6)
V_col = 10*1**2*pi*1e-6
Conc_molL = 5e-4

def plot_sim_results(simulation_results):
    plt.rcParams.update({'font.size': 11})
    c_out = simulation_results.solution.cond_det.outlet.total_concentration_components
    t_out = simulation_results.solution.cond_det.outlet.time
    
    plt.figure(figsize=(10,3))
    plt.plot(t_out/60, c_out[:, 0], lw=1.5, color='tab:blue')
    plt.plot(t_out/60, c_out[:, 1:]*1e5, lw=1.5, color='tab:red')
    
    plt.xlabel('Time [min]')
    plt.ylabel('Salt concentration [mM]\nProtein concentration [1e5 *mM]')

    A = mpatches.Patch(color='tab:blue', label='Simulated Salt')
    B = mpatches.Patch(color='tab:red', label='Simulated Protein')
    plt.legend(loc=0, handles=[A, B], fontsize=9)


def generate_poros_xs(component_system):
    binding_model = StericMassAction(component_system)
    binding_model.is_kinetic = False
    binding_model.adsorption_rate = [1e4, 1e4]
    binding_model.desorption_rate = [1, 1]
    binding_model.capacity = 180
    binding_model.steric_factor = [0, 50]
    binding_model.characteristic_charge = [1, 11.5]
    
    column = LumpedRateModelWithPores(component_system, name='column')
    column.binding_model = binding_model
    column.bed_porosity = 0.30
    column.particle_porosity = 0.8
    column.particle_radius = 50*1e-06 # 50 um
    column.film_diffusion = [1, 2.11618851e0] 
    column.length = 0.1
    column.diameter = 0.01
    column.axial_dispersion =  3.420580134283602e-09
    column.c = [10, 0]
    column.q = [binding_model.capacity, 0]

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


def generate_flowsheet():
    # Unit Operations
    component_system = ComponentSystem(['Salt', 'Protein'])
    column = generate_poros_xs(component_system)
    flowsheet = generate_akta(component_system, column=column)
    return flowsheet


def generate_process(flowsheet, elution_dur):
    # Timepoints of different events
    CV_time = ((flowsheet.column.diameter*0.5)**2*flowsheet.column.length*pi)/Q_m3s # this defines the duration of 1 CV
    StartOfElution = 0.6e-06/Q_m3s
    StartOfGradientDelay = StartOfElution + elution_dur*CV_time
    StartOfStrip = StartOfGradientDelay+ 5*CV_time
    
    # Gradient Elution Slope
    SaltGradientStart = 100 # mM
    SaltGradientEnd = 500 # mM
    
    concentration_difference = np.array([SaltGradientEnd, 0.0]) - np.array([SaltGradientStart, 0.0])
    gradient_duration = StartOfGradientDelay - StartOfElution
    gradient_slope = concentration_difference/gradient_duration
    
    process = Process(flowsheet, name='process_16cv')
    process.cycle_time = StartOfStrip + 5*CV_time
    process.add_event('inject', 'flow_sheet.inlet.c', [10, Conc_molL*1e3], time=0)
    process.add_event('grad_start', 'flow_sheet.inlet.c', [[100, gradient_slope[0]], [0, 0]], StartOfElution)
    process.add_event('grad_delay', 'flow_sheet.inlet.c', [500, 0], StartOfGradientDelay)
    process.add_event('strip', 'flow_sheet.inlet.c', [1000, 0], StartOfStrip) 

    return process


def obj_fun(x, reference_data):
    flowsheet = generate_flowsheet()
    simulator = Cadet()
    # assign variables for optimization
    flowsheet.column.binding_model.characteristic_charge = [1, x[0]]
    flowsheet.column.binding_model.adsorption_rate = [1, np.exp(x[1])]
    flowsheet.column.film_diffusion = [1e-04, np.exp(x[2])]

    error = []
    for key in reference_data:
        process = generate_process(flowsheet, key)
        sim_results = simulator.simulate(process)
        
        reftime = reference_data[key].iloc[:,0]*60
        refabsorption = reference_data[key].iloc[:,9]
        reference = ReferenceIO(f'{key}cv', reftime, refabsorption)
        comparator = Comparator(name='comparator_cv')
        comparator.add_reference(reference)
        comparator.add_difference_metric('Shape', reference, 'uv_det.outlet', components='Protein',)

        # calculate error for current dataset
        error.append(comparator(sim_results))

    # simplify error metric
    error = np.sum(error)
    print(x, error)
    return error