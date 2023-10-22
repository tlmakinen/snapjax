import matplotlib
import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.stats as ss


from astropy import units as u
from astropy.coordinates import SkyCoord

# selection effects function
eps,gmB,gc,gx1 = (29.96879778, -1.34334963,  0.45895811,  0.06703621)
selection_param=(gc, gx1, gmB, eps)

def log_indiv_selection_fn(phi_i, selection_param=np.array([gc, gx1, gmB, eps])):
    coefs = np.array(selection_param)
    position = np.concatenate([phi_i, np.ones(1)]) #np.array([*phi_i, 1])
    argument = np.dot(coefs, position)
    return ss.norm.logcdf(np.sqrt(np.pi/8)*argument) # must be a logcdf so it dies/grows to 0/1 at the right speed 



c_light = 299792.0 # Speed of light in km/s
h = 0.72 # Hubble parameter  # EDIT from 0.72
H_0 = 100.0 # Hubble constant
M_0 = -19.3 # Intrinsic magnitude
sn1a_variance = 0.16
cmb_variance = 0.04 ** 2.0
cmb_constraint = 1.3 # sum of matter and dark energy density
angular_diameter_constraint = 1408
DA_variance = 45.0 ** 2.0
hubbles_constant = 100


true_alpha = 0.13
true_beta = 2.56
intrinsic_dispersion = 0.1
true_omegam = 0.3
true_omegade= 0.7


true_x1_mean = 0
true_c_mean = 0
Rx = 1.0
Rc = 0.1


number_of_sne = int(sys.argv[8])


print("Dipole Value: ", sys.argv[1])
print("Quadrupole Value: ", sys.argv[2])
print("LC Params filename: ", sys.argv[3])
print("Full Covariance filename: ", sys.argv[4])
print("Distribution on sky: ", sys.argv[5])
print("Selected LC Params filename: ", sys.argv[6])
print("Selected Covariance filename: ", sys.argv[7])

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm



def get_norm_cartesian(l,b):
    cartesian_form = np.array([np.cos(b)*np.cos(l),np.cos(b)*np.sin(l),np.sin(b)])
    #print("Cartesian Form:",cartesian_form)
    cartesian_form = normalise(cartesian_form)
    return cartesian_form



#Dipole values
dipole_amplitude = float(sys.argv[1])
dipole_l = np.pi/12.0 #15 degrees
dipole_b = np.pi/6.0 #30 degrees
dipole_direction = get_norm_cartesian(dipole_l,dipole_b)


#Quadrupole values
quadrupole_amplitude = float(sys.argv[2])
quadrupole_l = np.pi/8.0 #22.5 degree
quadrupole_b = np.pi/4.0 #45 degrees
quadrupole_direction = get_norm_cartesian(quadrupole_l,quadrupole_b)




def get_dipole(dipole_amplitdue, dipole_direction , sn_direction):
    return (dipole_amplitude * np.dot(dipole_direction, sn_direction))


def get_quadrupole(quadrupole_amplitude, quadrupole_direction, sn_direction):
    return (quadrupole_amplitude * ((3.0 * (np.dot(quadrupole_direction,sn_direction) ** 2.0)) - 1.0)) #m=0 l=2


def generate_covariance_matrix(data):
    cov_data = np.zeros((np.shape(data)[0]*3,np.shape(data)[0]*3)) # Assume diagonal covariance
    for i in range(0,np.shape(data)[0]):
        cov_data[i*3][i*3] = data[i][6] ** 2.0 # Set c value
        cov_data[i*3 + 1][i*3 + 1] = data[i][4] ** 2.0# Set x value
        cov_data[i*3 + 2][i*3 + 2] = data[i][2] ** 2.0# Set mb value
    return cov_data



def get_magnitude(mu, M, alpha, x1, beta, colour):
    return mu + M - (alpha * x1) + (beta * colour)


def distance_modulus(z, matter_density,d_energy_density, sn_l, sn_b):
    #print(type(z),type(matter_density),type(d_energy_density),type(c_light),type(h))
    sn_cartesian = get_norm_cartesian(sn_l,sn_b)
    eta = (-5.0 * np.log10(H_0*h/c_light)) + 25.0
    #print(eta)
    #print(z,matter_density,d_energy_density)
    theoretical_distance_modulus = eta + (5*np.log10(luminosity_distance(z, matter_density,d_energy_density)))
    #introduce dipole/quadrupole here
    dipole_value = get_dipole(dipole_amplitude,dipole_direction,sn_cartesian)


    quadrupole_value = get_quadrupole(quadrupole_amplitude,quadrupole_direction,sn_cartesian)
    #print("Quadrupole: ",quadrupole_value)
    #print("Dipole value:",dipole_value,sn_cartesian,sn_l,sn_b)
    theoretical_distance_modulus = theoretical_distance_modulus * (1 + dipole_value + quadrupole_value)

    return theoretical_distance_modulus




def luminosity_distance(z, matter_density, d_energy_density):
    '''
    An integral to be evaluated based on the redshift value and density parameters
    '''
    curvature_density = 1.0 - matter_density - d_energy_density
    #print(curvature_density, matter_density, d_energy_density)
    def luminosity_integrand(z):
        '''
        The integrand part of the luminosity_distance
        '''
        matter_term = matter_density * ((1.0 + z)**3.0)
        energy_term = d_energy_density
        curvature_term = curvature_density * ((1.0 + z)**2.0)
        integrand = (matter_term + energy_term + curvature_term)**(-0.5)
        return integrand

    integrand,err = quad(luminosity_integrand,0.0,z)


    l_distance = np.sqrt(np.abs(curvature_density)) * integrand
    #define front term within the conditions to avoid the divide by zero error.
    if(curvature_density>0):
        front_term = ((1.0 + z)/(np.sqrt(np.abs(curvature_density))))
        l_distance =  front_term * np.sinh(l_distance)
    elif(curvature_density <0):
        front_term = ((1.0 + z)/(np.sqrt(np.abs(curvature_density))))
        l_distance = front_term * np.sin(l_distance)
    elif(curvature_density == 0):
        #We don't use front_term * l_distance here due to the divide by zero error,
        #instead we use the simplified equation that appears when curvature is 0.
        l_distance = (1 + z) * integrand



    #print(l_distance)sc
    return l_distance



#####DATA
snarray = []
with open('jla_lcparams.txt') as f:
    for line in f:
        snarray.append([(x) for x in line.split()])
'''
name = [snarray[i][0] for i in range(1,240)]
Zcmb = [float(snarray[i][1]) for i in range(1,240)]
Zhel = [float(snarray[i][2]) for i in range(1,240)]
mb = [float(snarray[i][4]) for i in range(1,240)]
dmb = [float(snarray[i][5]) for i in range(1,240)]
x1 = [float(snarray[i][6]) for i in range(1,240)]
dx1 = [float(snarray[i][7]) for i in range(1,240)]
c = [float(snarray[i][8]) for i in range(1,240)]
dc = [float(snarray[i][9]) for i in range(1,240)]
'''


name = [snarray[i][0] for i in range(1,len(snarray))]
Zcmb = [float(snarray[i][1]) for i in range(1,len(snarray))]
Zhel = [float(snarray[i][2]) for i in range(1,len(snarray))]
mb = [float(snarray[i][4]) for i in range(1,len(snarray))]
dmb = [float(snarray[i][5]) for i in range(1,len(snarray))]
x1 = [float(snarray[i][6]) for i in range(1,len(snarray))]
dx1 = [float(snarray[i][7]) for i in range(1,len(snarray))]
c = [float(snarray[i][8]) for i in range(1,len(snarray))]
dc = [float(snarray[i][9]) for i in range(1,len(snarray))]
ra = [float(snarray[i][16]) for i in range(1,len(snarray))]
dec = [float(snarray[i][17]) for i in range(1,len(snarray))]


#Generate KDE estimate of position DATA
values = np.vstack([ra, dec])
kernel = ss.gaussian_kde(values)


xmin = min(ra)
xmax = max(ra)
ymin = min(dec)
ymax = max(dec)
print(xmin,xmax,ymin,ymax)



X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)




Zcmb_mean = np.mean(Zcmb)
Zcmb_std = np.std(Zcmb)


dmb_mean = np.mean(dmb)
dmb_std = np.std(dmb)


dx1_mean = np.mean(dx1)
dx1_std = np.std(dx1)


dc_mean = np.mean(dc)
dc_std = np.std(dc)


mb_mean = np.mean(mb)
mb_std = np.std(mb)


x1_mean = np.mean(x1)
x1_std = np.std(x1)


c_mean = np.mean(c)
c_std = np.std(c)

#print(Zcmb_mean, Zcmb_std, dmb_mean, dmb_std, dx1_mean, dx1_std, dc_mean, dc_std)





def generate_sn1a():

    #Steps followed in March et al 20011

    #Step, 1 draw latent redshift
    #z = np.random.normal(loc=Zcmb_mean, scale = Zcmb_std)
    q = ss.powerlaw.rvs(2.5, loc=0, scale=2.3) # q = z+1 from Dilday et al
    while q-1 < 0:
        q = ss.powerlaw.rvs(2.5, loc=0, scale=2.3)
    z = q-1
    l = 0
    b = 0
    valid_coordinate = False
    while(not valid_coordinate):
        try:
            if(sys.argv[5] == "kde"):
                #Generate galactic coordinateds longitude(l) and latitude (b)
                position_sample = kernel.resample(1)
                #print("position: " , position_sample[0][0], position_sample[1][0])
                pos_ra = position_sample[0][0]
                pos_dec = position_sample[1][0]
                c = SkyCoord(ra = pos_ra, dec =pos_dec, unit = (u.degree,u.degree))
                galactic_coordinates = c.galactic
                l = galactic_coordinates.l.rad
                b = galactic_coordinates.b.rad
                valid_coordinate = True
            elif(sys.argv[5]=="uniform"):
                l = np.random.uniform(0.0,2.0*np.pi)
                b = np.random.uniform(-np.pi/2.0,np.pi/2.0)
                valid_coordinate = True
            else:
                print("Not enough arguments or position distribution type not specified")
                sys.exit()
        except Exception as e:
            print("Invalid coordinate sampled")
            print(e)
    while(z<0):
        #resample - negative redshift sampled
        q = ss.powerlaw.rvs(1.5, loc=1.0, scale=3.0, size=(1000,)) # x = z+1 from Dilday et al
        z = q-1
        #z = np.random.normal(loc=Zcmb_mean, scale = Zcmb_std)
    #Step 2, compute mu_i using fiducial values
    mu_i = distance_modulus(z,true_omegam,true_omegade,l,b)


    #Step 3, Draw latent x1_i, c_i, M_i
    M_i = np.random.normal(loc=M_0, scale = intrinsic_dispersion)
    x1_i = np.random.normal(loc=true_x1_mean, scale = Rx)
    c_i = np.random.normal(loc=true_c_mean, scale = Rc)


    #Step 4, compute exact mb* from distance modulus
    mb = get_magnitude(mu_i,M_i, true_alpha, x1_i, true_beta, c_i)

    #Step 5, draw standard deviations of x,c and mb from survey data. These could be negative, temporarily handled by turning positive.
    observed_dx1 = np.random.normal(loc=dx1_mean, scale = dx1_std)
    while (observed_dx1 < 0):
        #negative error so resample
        observed_dx1 = np.random.normal(loc=dx1_mean, scale = dx1_std)


    observed_dc1 = np.random.normal(loc=dc_mean, scale = dc_std)
    while (observed_dc1 < 0):
        observed_dc1 = np.random.normal(loc=dc_mean, scale = dc_std)


    observed_dmb = np.random.normal(loc=dmb_mean, scale = dmb_std)
    while(observed_dmb < 0):
        observed_dmb = np.random.normal(loc=dmb_mean, scale = dmb_std)


    #Step 6, Draw observed SALT-II values of x,c and mb
    observed_mb = np.random.normal(loc=mb, scale = observed_dmb)
    observed_x1 = np.random.normal(loc = x1_i, scale = observed_dx1)
    observed_c1 = np.random.normal(loc = c_i, scale = observed_dc1)
    true_mb = mb
    true_x1 = x1_i
    true_c = c_i

    #Step 7: Apply DES-like Selection cuts on OBSERVED values
    phi_i = np.array([observed_c1, observed_dx1, observed_mb])
    selection_prob_i = np.exp(log_indiv_selection_fn(phi_i))
    selection_tag = 0

    # draw random number to simulate whack chance of SN1a being missed
    rand = np.random.uniform(low=0, high=1.0)

    if selection_prob_i > rand:
        selection_tag  = 1
    else:
        selection_tag = 0



    return (z,observed_mb,observed_dmb,observed_x1,observed_dx1,observed_c1,observed_dc1,l,b,true_mb,true_x1,true_c, selection_prob_i, selection_tag)







sim_z = []
sim_zhel = []
sim_mb = []
sim_dmb = []
sim_x1 = []
sim_dx1 = []
sim_c = []
sim_dc = []
sim_l = []
sim_b = []
sim_true_mb = []
sim_true_x1 = []
sim_true_c = []
sim_selection_prob = []
sim_selection_tag = []

# Create DUMP file with all generated (seen and unseen) SN1a
num_observed = 0

while num_observed < number_of_sne:
    sim = generate_sn1a()
    sim_z.append(sim[0])
    sim_zhel.append(sim[0])
    sim_mb.append(sim[1])
    sim_dmb.append(sim[2])
    sim_x1.append(sim[3])
    sim_dx1.append(sim[4])
    sim_c.append(sim[5])
    sim_dc.append(sim[6])
    sim_l.append(sim[7])
    sim_b.append(sim[8])
    sim_true_mb.append(sim[9])
    sim_true_x1.append(sim[10])
    sim_true_c.append(sim[11])
    sim_selection_prob.append(sim[12])
    sim_selection_tag.append(sim[13])
    if sim[13] > 0:
        num_observed += 1


sim_z = np.array(sim_z)
sim_zhel = np.array(sim_zhel)
sim_mb = np.array(sim_mb)
sim_dmb = np.array(sim_dmb)
sim_x1 = np.array(sim_x1)
sim_dx1 = np.array(sim_dx1)
sim_c = np.array(sim_c)
sim_dc = np.array(sim_dc)
sim_l = np.array(sim_l)
sim_b = np.array(sim_b)
sim_true_mb = np.array(sim_true_mb)
sim_true_x1 = np.array(sim_true_x1)
sim_true_c = np.array(sim_true_c)
sim_selection_prob = np.array(sim_selection_prob)
sim_selection_tag = np.array(sim_selection_tag)



# for selected SN1a
all_simulated_data = (np.array([sim_z,sim_mb,sim_dmb,sim_x1,sim_dx1,sim_c,sim_dc,sim_l,sim_b,sim_true_mb,sim_true_x1,sim_true_c, sim_selection_prob, sim_selection_tag]).transpose())
mask = (all_simulated_data[:, 13] > 0).astype(bool)
miss_mask = np.invert(mask)

fig = plt.figure(figsize=(20,10))
fig.suptitle('Simulated SN1a data from JLA full sample')
plt.subplot(241)
plt.xlabel('$z$')
plt.ylabel('$m_{B}$')
plt.scatter(sim_z,sim_mb,marker = 'x',s=5, label="Simulated")
plt.scatter(Zcmb,mb,marker = 'x',s=5, label="JLA")
plt.scatter(np.array(sim_z)[mask], np.array(sim_mb)[mask], marker='+', color='k', s=4, label='Selected')
plt.ylim(bottom = 15)
plt.xlim(left = 0)
plt.legend(bbox_to_anchor=(0,1.2), loc="upper left")


plt.subplot(242)
plt.xlabel('$z$')
plt.ylabel('$x_{1}$')
plt.scatter(sim_z,sim_x1,marker = 'x',s=5)
plt.scatter(Zcmb,x1,marker = 'x',s=5)
plt.scatter(np.array(sim_z)[mask], np.array(sim_x1)[mask], marker='+', color='k', s=4)
plt.ylim(bottom = -5)
plt.xlim(left = 0)


plt.subplot(243)
plt.xlabel('$z$')
plt.ylabel('c (colour)')
plt.scatter(sim_z,sim_c,marker = 'x',s=5)
plt.scatter(Zcmb,c,marker = 'x',s=5)
plt.scatter(np.array(sim_z)[mask], np.array(sim_c)[mask], marker='+', color='k', s=4)
plt.ylim(bottom = -0.5)
plt.xlim(left = 0)


plt.subplot(244)
plt.xlabel('$\sigma_{c}$')
plt.ylabel('$\sigma_{x_{1}}$')
plt.scatter(sim_dc,sim_dx1,marker = 'x',s=5)
plt.scatter(dc,dx1,marker = 'x',s=5)
plt.ylim(bottom = 0)
plt.xlim(left = 0)


plt.subplot(245)
plt.xlabel('$\sigma_{x_{1}}$')
plt.ylabel('$\sigma_{m_{B}}$')
plt.scatter(sim_dx1,sim_dmb,marker = 'x',s=5)
plt.scatter(dx1,dmb,marker = 'x',s=5)
plt.ylim(bottom = 0)
plt.xlim(left = 0)


plt.subplot(246)
plt.xlabel('$\sigma_{c}$')
plt.ylabel('$\sigma_{m_{B}}$')
plt.scatter(sim_dc,sim_dmb,marker = 'x',s=5)
plt.scatter(dc,dmb,marker = 'x',s=5)
plt.ylim(bottom = 0)
plt.xlim(left = 0)


#Convert JLA coordinates to galatic coordinates
c = SkyCoord(ra = ra, dec = dec, unit = (u.degree,u.degree))
l = c.galactic.l.rad
b = c.galactic.b.rad
plt.subplot(247)
plt.xlabel('$l$')
plt.ylabel('$b$')
plt.scatter(sim_l,sim_b,marker = 'x',s=5)
plt.scatter(l,b,marker = 'x',s=5)


plt.savefig('simulatedsne.png')

# plot selection function for selected vs missed SN1a
fig = plt.figure(figsize=(24,8))
fig.suptitle('JLA Simulated SN1a Selection Probability')
# selection fn in mB
plt.subplot(141)
plt.xlabel('$m_B$')
plt.ylabel('$p$(selection)')
plt.scatter(np.array(sim_mb)[miss_mask], np.array(sim_selection_prob)[miss_mask], 
                    color='r', marker ='.', s=3, label='{} Missed SN1a'.format(np.sum(miss_mask)))
plt.scatter(np.array(sim_mb)[mask], np.array(sim_selection_prob)[mask], 
                    color='b', marker ='.', s=3, label='{} Selected SN1a'.format(np.sum(mask)))
plt.legend(loc="lower left")

# for c
plt.subplot(142)
plt.xlabel('$c$')
plt.ylabel('$p$(selection)')
plt.scatter(np.array(sim_c)[miss_mask], np.array(sim_selection_prob)[miss_mask], 
                    color='r', marker ='.', s=3)
plt.scatter(np.array(sim_c)[mask], np.array(sim_selection_prob)[mask], 
                    color='b', marker ='.', s=3)

# for x1
plt.subplot(143)
plt.xlabel('$x_1$')
plt.ylabel('$p$(selection)')
plt.scatter(np.array(sim_x1)[miss_mask], np.array(sim_selection_prob)[miss_mask], 
                    color='r', marker ='.', s=3)
plt.scatter(np.array(sim_x1)[mask], np.array(sim_selection_prob)[mask], 
                    color='b', marker ='.', s=3)

# for z
plt.subplot(144)
plt.xlabel('$z$')
plt.ylabel('$p$(selection)')
plt.scatter(np.array(sim_z)[miss_mask], np.array(sim_selection_prob)[miss_mask], 
                    color='r', marker ='.', s=3)
plt.scatter(np.array(sim_z)[mask], np.array(sim_selection_prob)[mask], 
                    color='b', marker ='.', s=3)

plt.savefig('simuated_selection_fn.png')
#plt.show()


headers = "z mb dmb x1 dx1 c dc l b true_mb true_x1 true_c"
#all_simulated_data = (np.array([sim_z,sim_mb,sim_dmb,sim_x1,sim_dx1,sim_c,sim_dc,sim_l,sim_b,sim_true_mb,sim_true_x1,sim_true_c, sim_selection_prob, sim_selection_tag]).transpose())

# now take the slice of DUMP file where the SN1a are observed
mask = (all_simulated_data[:, 13] > 0).astype(bool)

# make dump file for all data
simulated_data = (np.array([sim_z,sim_mb,sim_dmb,sim_x1,sim_dx1,sim_c,sim_dc,sim_l,sim_b,sim_true_mb,sim_true_x1,sim_true_c]).transpose())
simulated_covariance = generate_covariance_matrix(simulated_data)
# make selected file for inference
selected_simulated_data = simulated_data[mask]
selected_covariance = generate_covariance_matrix(selected_simulated_data)


np.savetxt(sys.argv[3],simulated_data, delimiter=' ',header = headers,comments="")
np.savetxt(sys.argv[4], simulated_covariance, delimiter=' ')

np.savetxt(sys.argv[6],selected_simulated_data, delimiter=' ',header = headers,comments="")
np.savetxt(sys.argv[7], selected_covariance, delimiter=' ')
