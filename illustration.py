from mpacts.commands.force.body import CentralPullCommand
from mpacts.commands.force.body import CentrifugalTorqueCommand
from mpacts.commands.force.body import LinearStretchingTorqueCommand
from mpacts.commands.force.constraints.virtualforces import AddNodeForcesToSegmentForcesCommand
from mpacts.commands.force.constraints.virtualforces import AddNodeForcesToSegmentTorqueCommand
from mpacts.commands.force.random import BrownianTranslationForcesCommand
from mpacts.commands.force.random import BrownianRotationalForcesCommand

from mpacts.commands.geometry.cylinder import ComputeCylinderIndexListCommand

from mpacts.commands.misc.frictionmatrices import AssembleCylinderFrictionCoefficientsCommand
from mpacts.commands.misc.frictionmatrices import AssembleCylinderRotationalFrictionCoefficientsCommand

from mpacts.commands.monitors.progress import ProgressIndicator

from mpacts.commands.time_evolution.integration import ForwardEuler_Generic
from mpacts.commands.time_evolution.integration import ComputeVelocitiesFromSegmentForcesAndTorque 

from mpacts.contact.detectors.multigrid import MultiGridContactDetector

from mpacts.contact.models.collision.hertz.hertz_capsulewithtorque import HertzModelForCapsulesWithTorque

from mpacts.core.arrays import create_array
from mpacts.core.command import ExecuteOnce
from mpacts.core.command import ExecuteTimeInterval
from mpacts.core.configuration import parallel_configuration
from mpacts.core.simulation import Simulation
from mpacts.core.units import unit_registry as u
from mpacts.core.valueproperties import Variable, VariableFunction

from mpacts.particles.deformablebody import DeformableBody
from mpacts.particles.deformablecapsulewithrotations import DeformableCapsuleWithRotations
from mpacts.particles.node import Node
from mpacts.particles.particlecontainer import ParticleContainer

from mpacts.tools.setcontactdatastorage import SetContactDataStorage
from mpacts.tools.random_seed import set_random_seed

import numpy as np 

#--------------------------------------------------------------
# Global settings
#--------------------------------------------------------------
rng_seed = 814559961    # seed used for all (internal, numpy, python.random) PRNG generators
thread_count = 12       # number of threads when multithreaded  
grainsize = 128         # number of contact per single thread when multithreaded

set_random_seed( rng_seed )
SetContactDataStorage( "Vector" )

try:
    parallel_configuration().set(parallel_grainsize = grainsize )
    parallel_configuration().set(n_threads = thread_count ) 
except:
    pass


#--------------------------------------------------------------
# Simulation and global constants 
#--------------------------------------------------------------
dt = 0.1 * u("ms")
snapshotevery = 0.1 * u("s")  
endTime = 1 * u("min")  

sim = Simulation("Simulation", timestep = dt )
params = sim("params") 

#Properties of environment
eta = Variable("eta", params, value = 0.79e-3 * u("Pa * s"), description = "Viscosity of the environment. [Pa.s]" )   #Water at 30 C
Variable("k_B", params, value = 1.38e-23 * u("J/K"), description = "Boltzmann constant. [J/K]" )                
Variable("T", params, value = 303.15 * u("K"), description = "Temperature of the environment. [K]" )            #30 C
kBT = VariableFunction("kBT", params, function = "$T$ * $k_B$", description = "Temperature of environment in Joules. [J]" )

#Timestep
dt_var = Variable("dt", params, value = sim.get_property("timestep"), description = "Simulation timestep. [s]" ) 
#Contact detection refresh rate - how often is done the full contact detection
freq = Variable("update_every", params, value = 100, description = "Frequency of contact detection. [number of timesteps]" )

#Friction constants for cylinders - see Broesma DOI: 10.1063/1.441071, Li & Tang DOI: 10.1103/PhysRevE.69.061921
c_para = Variable("c_friction_para", params, value = -0.114, description = "Correction coefficient for the drag of the polymer, parallel component. [1]" )
c_perp = Variable("c_friction_perp", params, value = 0.886, description = "Correction coefficient for the drag of the polymer, perpendicular component. [1]" )
c_rot  = Variable("c_friction_rot", params, value = -0.447, description = "Correction coefficient for the drag of the polymer, rotational component component. [1]" )

#Settings of the setup 
c = 0.05 * u("1/um^3") # Concentration of bacteria - cubic latice is approx 0.2 
R = 25 * u("um") #Radius of the colony 

#Number of bacteria is drawn from poisson distribution
N = np.random.poisson( 4./3.*np.pi*R**3*c ) 

#Minimal and maximal radius and length 
r_min = 0.5 * u("um")
r_max = 1. * u("um") 
l_min = 2. * u("um")
l_max = 5. * u("um")

#--------------------------------------------------------------
# Bacteria properties
#--------------------------------------------------------------
#Elastic properies of bacteria
tau = Variable("bacteria_tau", params, value = 100. * u("ms"), description = "Relaxation time of longitudinal deformations. Has to be much bigger than dt." )
#Artificially decreased - real value is around 300 kPa 
E = Variable("bacteria_Youngs_modulus", params, value = 3 * u("kPa"), description = "Stiffness of bacteria." ) 
nu = Variable("bacteria_Poisson_ratio", params, value = 0, description = "Poisson ratio of bacteria." ) 

#Attractive force towards the center
F_attr = Variable("bacteria_attraction_magnitude", params, value = 20 * u("fN"), description = "Magnitude of the attractive force towards the origin.") 

#On the level of 4 sigma 
keep_distance_bacteria = VariableFunction("bacteria_bacteria_keep_distance", params, function = " 4 * math.sqrt( 6. * $kBT$ * $dt$ * $update_every$ / ( 2. * math.pi * $eta$ * %g / ( math.log( 2. * %g / %g ) + $c_friction_para$ ) ) ) " % (l_min.to("m").magnitude, l_min.to("m").magnitude, r_min.to("m").magnitude ), description = "Estimate on keep distance based on diffusion on the level of 4 sigma." )

#--------------------------------------------------------------
# Bacteria
#  - here we define the model for bacteria
#--------------------------------------------------------------
print "Setting up model ... "

#Bacteria are represented as deformable capsules with overdamped dynamics
bacteria = ParticleContainer("bacteria", DeformableBody.compose( ( Node, "nodes"), ( DeformableCapsuleWithRotations, "segments", "nodes") ), sim)

#We create an array for equilibrium length and stiffness of each bacteria
bacteria("segments").create_array("Scalar", "equilibrium_length") 
bacteria("segments").create_array("Scalar", "stiffness") 

#Array for containing the IDs of segments which the given node is part of
bacteria("nodes").create_array("IndexVector", "nodeIndexList")
#and a command to fill it - as we do not add or destroy elements we need to run it only once 
ComputeCylinderIndexListCommand("bacteria_nodes_get_segment_indices", sim, pc = bacteria("segments"), nodeIndexList = bacteria("nodes")["nodeIndexList"], gate = ExecuteOnce() )

#Commands for obtaining the diffusion properties 
AssembleCylinderFrictionCoefficientsCommand("bacteria_friction_trans_coef", sim, pc = bacteria("segments"), c_para = c_para, c_perp = c_perp, eta = eta) 
AssembleCylinderRotationalFrictionCoefficientsCommand("bacteria_friction_rot_coef", sim, pc = bacteria("segments"), c_rot = c_rot, eta = eta, k_array = bacteria("segments")["stiffness"], kBT = kBT, tau = tau, l0_array = bacteria("segments")["equilibrium_length"] ) 

#The diffusion forces and torques computation itself
BrownianTranslationForcesCommand("bacteria_thermal_translations", sim, sim = sim, pc = bacteria("segments"), temperature = kBT )
BrownianRotationalForcesCommand("bacteria_thermal_rotations", sim, sim = sim, pc = bacteria("segments"), temperature = kBT )

CentrifugalTorqueCommand("bacteria_centrifugal", sim, pc = bacteria("segments"), temperature = kBT )

#Time integration
ComputeVelocitiesFromSegmentForcesAndTorque("bacteria_get_velocities", sim, pc = bacteria('segments'))
ForwardEuler_Generic("bacteria_velocity_integration", sim, pc = bacteria("nodes"), x = bacteria("nodes")["x"], dx = bacteria("nodes")["v"] )

#Elasticity of the model 
LinearStretchingTorqueCommand("bacteria_elasticity", sim, pc = bacteria("segments"), k_array = bacteria("segments")["stiffness"], l0_array = bacteria("segments")["equilibrium_length"] )

#Repulsion
model_repulsion = HertzModelForCapsulesWithTorque("bacteria_repulsion", sim, pc1 = bacteria("segments"), pc2 = bacteria("segments"), E1 = E, E2 = E, nu1 = nu, nu2 = nu )
MultiGridContactDetector("bacteria_repulsion_cd", sim, cmodel = model_repulsion, keep_distance = keep_distance_bacteria, update_every = freq )

#Attractive potential towards the center
CentralPullCommand("bacteria_attraction", sim, pc = bacteria("nodes"), force = F_attr )
#Because we add those forces on nodes (we need position for that command) we need to collect it to the segment
AddNodeForcesToSegmentForcesCommand("bacteria_transfeor_node_forces_to_segment_force", sim, pc = bacteria("segments") )
AddNodeForcesToSegmentTorqueCommand("bacteria_transfeor_node_forces_to_segment_torque", sim, pc = bacteria("segments") )


#Writing out all data
writer = bacteria.VTKWriterCmd( gate = ExecuteTimeInterval( sim = sim, interval = snapshotevery.to("s").magnitude ), directory = "./" ) 
writer.select_all(True) 

#--------------------------------------------------------------
# Seeding the bacteria 
#--------------------------------------------------------------
print "Setting up %d bacteria ..." % N

#Generate proportions of bacteria
proportions = np.random.uniform(0,1,N) 
lengths = ( l_max - l_min ) * proportions + l_min
radiuses = ( r_max - r_min ) * proportions + r_min

#Generate positions uniformly in a sphere 
# First generate radia 
r =  np.sqrt( (R-radiuses)**2. - 0.25 * lengths**2 ) * np.random.uniform(0,1,N)**(1./3.) 

#Based on the radius compute possible orientations of individual bacteria and pick up a random one 
phi = np.random.uniform( 0.,2.*np.pi, N)

max_cos_th = ( ( (R-radiuses)**2 - r**2 - 0.25 * lengths**2 )/( r * lengths ) ).to("1") 
max_cos_th = np.maximum( -1., np.minimum( 1., max_cos_th ) ) 

cos_th = max_cos_th * np.random.uniform( -1.,1.,N )
sin_th = np.sqrt( 1. - cos_th**2 ) 

#Positions and orientations  
t = np.array( [ sin_th * np.cos(phi) , sin_th * np.sin(phi), cos_th ] )

#Redistribute on the surface of sphere 
#Generate new angles 
pos_phi = np.random.uniform(0,2.*np.pi,N)
pos_cos_theta = np.random.uniform(-1.,1,N)

pos_sin_theta = np.sqrt( 1 - pos_cos_theta**2 )

#Unit vector in the direction of the position
pos_u = np.array( [ pos_sin_theta * np.cos( pos_phi ), pos_sin_theta * np.sin( pos_phi ), pos_cos_theta ] )

#Vector to get the rotation around 
omega = map( lambda x : np.cross( [0,0,1], x ) , pos_u.T ) 
omega = map( lambda x : x / np.linalg.norm( x ), omega )

#Full rotational matrix 
R = np.array( [ np.eye(3) * np.dot( [0,0,1] , pos_u[:,i] ) + ( 1. - np.dot( [0,0,1] , pos_u[:,i] ) ) * np.outer( omega[i], omega[i] ) + np.outer( pos_u[:,i], [0,0,1] ) - np.outer( [0,0,1], pos_u[:,i] ) for i in range(N) ] )

#New positions and orientations 
x = ( pos_u * r ).T
t = np.einsum( "ijk,ki->ij", R, t )  

print "Adding bacteria ... " 

#Add particles 
for i in range(N) :
    p = bacteria.add_particle()
    #print "%.01f\r" % (100*(i + 0.) / N), 

    #Positions of points 
    px = [ x[i] + 0.5 * t[i] * lengths[i], x[i] - 0.5 * t[i] * lengths[i] ] 

    #Stiffness from Hooks law
    k = ( E.get_property("value") * u("Pa") * np.pi * radiuses[i]**2 / lengths[i] ).to("N/m").magnitude  

    #Add end points
    p.nodes.add_and_set_particles( x = px, r = radiuses[i] )

    #Add segment connecting them 
    p.segments.add_and_set_particles(   r = [ radiuses[i] ], 
                                        t = tuple( t[i] ),
                                        vertexIndices = [(0,1)],
                                        equilibrium_length = lengths[i], 
                                        l = lengths[i] ,
                                        gamma = (0.,0.,0.),
                                        gamma_rot = (0.,0.,0.),
                                        stiffness = k 
                                    ) 

#--------------------------------------------------------------
#Simulation itself
#--------------------------------------------------------------
ProgressIndicator("PrintProgress", sim, print_interval=5)

#Relaxation with smaller timestep first
sim.set( timestep = 1 * u("ns" ) )
dt_var.set_property("value", 1 * u("ns" ) )
#sim.set_property("timestep", 1 * u("us" ) )
print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
sim.run_until( ( 100 * u("us") ).to("s").magnitude ) 

#Relaxation with smaller timestep first
sim.set( timestep = 10 * u("ns" ) )
dt_var.set_property( "value", 10 * u("ns" ) )
print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
sim.run_until( ( 1 * u("ms") ).to("s").magnitude ) 

#Relaxation with smaller timestep first
sim.set( timestep = 100 * u("ns" ) )
dt_var.set_property( "value", 100 * u("ns" ) )
print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
sim.run_until( ( 10 * u("ms") ).to("s").magnitude ) 

#Relaxation with smaller timestep first
sim.set( timestep = 1 * u("us" ) )
dt_var.set_property( "value", 1 * u("us" ) )
print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
sim.run_until( ( 100 * u("ms") ).to("s").magnitude ) 

#Relaxation with smaller timestep first
sim.set( timestep = 10 * u("us" ) )
dt_var.set_property( "value", 10 * u("us" ) )
print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
sim.run_until( ( 1 * u("s") ).to("s").magnitude ) 

#Relaxation with smaller timestep first
#sim.set( timestep = 100 * u("us" ) )
#dt_var.set_property( "value", 100 * u("us" ) )
#print "Relaxation with dt = %g s " % (sim.get_property("timestep"))  
#sim.run_until( ( 10 * u("s") ).to("s").magnitude ) 

print "Simulation start ..." 
#sim.set( timestep = dt )
#dt_var.set_property( "value", dt )
sim.run_until( endTime.to("s").magnitude ) 
