
import numpy as np

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
import topfarm

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018, Blondel_Cathelain_2020
from py_wake.utils.gradients import autograd
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.shear import PowerShear
import pickle


# with open('boundary_layouts_ENGIN480\utm_boundary2.pkl', 'rb') as f:
#     boundary = np.array(pickle.load(f))


# with open('/Users/hamadhaidar/Documents/software/boundary_layouts_ENGIN480/utm_boundary2.pkl', 'rb') as f:
#     boundary = np.array(pickle.load(f))


# with open('boundary_layouts_ENGIN480/utm_boundary2.pkl', 'rb') as f:
#     boundary = np.array(pickle.load(f))

# boundary_layouts_ENGIN480/utm_boundary2.pkl

# /Users/hamadhaidar/Documents/software/boundary_layouts_ENGIN480/utm_boundary2.pkl\
# boundary_layouts_ENGIN480/utm_boundary2.pkl

# with open('boundary_layouts_ENGIN480\utm_layout2.pkl', 'rb') as f:
#     xinit,yinit = np.array(pickle.load(f))

with open('boundary_layouts_ENGIN480/utm_boundary3D.pkl', 'rb') as f:
    # boundary = pickle.load(f)
    boundary = np.array(pickle.load(f))

    

with open('boundary_layouts_ENGIN480/utm_layout3D.pkl', 'rb') as f:
    xinit,yinit = np.array(pickle.load(f))

    # boundary = pickle.load(f)




maxiter = 1000
tol = 1e-6

# Sea Impact Website: adjust the power_norm (given in kW, for example, in this example here 11,000kW or 11MW), 
# adjust the diameter, adjust the hub_height 
class SG_8_0_167DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG_8_0_167DD', diameter=167, hub_height=150,
                             power_norm=8000, turbulence_intensity=0.14)

# Canvas Week 13: Global_Wind_Atlas.pdf
class dieppe(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.8716, 7.8686, 7.3079, 5.7191, 5.0873, 4.7004, 8.8319, 12.4057, 16.2378, 15.1056, 6.7622, 5.1018] 
        a = [8.51, 10.00, 8.81, 8.42, 9.09, 8.97, 13.41, 13.55, 12.75, 10.32, 8.87, 8.75] 
        k = [22.084, 2.451, 2.330, 2.201, 2.518, 1.955, 2.525, 2.924, 2.459, 2.143, 1.900, 1.807] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "dieppe"


wind_turbines = SG_8_0_167DD()

site = dieppe()

sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

def aep_func(x,y):
    aep = sim_res(x,y).aep().sum()
    return aep

def daep_func(x,y):
    daep = sim_res.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'], x=x, y=y)
    return daep

boundary_closed = np.vstack([boundary, boundary[0]])


cost_comp = CostModelComponent(input_keys=['x', 'y'],
                                          n_wt = len(xinit),
                                          cost_function = aep_func,
                                          cost_gradient_function=daep_func,
                                          objective=True,
                                          maximize=True,
                                          output_keys=[('AEP', 0)]
                                          )


problem = TopFarmProblem(design_vars= {'x': xinit, 'y': yinit},
                         constraints=[XYBoundaryConstraint(boundary, 'polygon'),
                                      SpacingConstraint(334)],
                        cost_comp=cost_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                        n_wt=len(xinit),
                        expected_cost=0.001,
                        plot_comp=XYPlotComp()
                        )


cost, state, recorder = problem.optimize()

#change here to name it differently according to each of the sites you're simulationg
recorder.save('optimization_dieppe.py')
# recorder.save('optimization_costalvirginia')


from topfarm.recorders import TopFarmListRecorder
import matplotlib.pyplot as plt

# recorder = TopFarmListRecorder().load('/Users/rafaelvalottarodrigues/Documents/software/farm_to_farm_bench/recordings/optimization_viveyard.pkl')
recorder = TopFarmListRecorder().load('/Users/rafaelvalottarodrigues/Documents/software/farm_to_farm_bench/recordings/optimization_borselle.pkl')


plt.figure()
plt.plot(recorder['counter'], recorder['AEP']/recorder['AEP'][-1])
plt.xlabel('Iterations')
plt.ylabel('AEP/AEP_opt')
plt.show()

print('donme')

print('done')

print('done')