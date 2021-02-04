import numpy as np

def vib_analysis(molecule,hessian):

    #input 
    #molecule: psi4 molecule class,
    #the mass(unit: amu) and coordinates(unit: angstrom) of each atoms in the molecule
    #hessian: (3*natom,3*natom) numpy 2d array, 
    #hessian matrix(unit: hartree/bohr**2)

    #output
    #freq: 1d numpy array 
    #normal mode vibrational frequency(unit: cm**(-1)): ()
    #disp_normal: (3*natom,3*natom-6) numpy 2d array,
    #displacement vectors for normal vibrational mode, normalized mass weight coordinate 

    #initialize variables
    speed_of_light = 299792458
    hartree_to_joule = 4.35974e-18
    bohr_to_meter = 5.2918e-11
    amu_to_kg = 1.66054e-27
    unit_conversion_factor = hartree_to_joule/(bohr_to_meter)**2/amu_to_kg

    natom = molecule.natom()
    coordinates = np.zeros((natom,3))
    cm_coordinates = np.zeros((natom,3))
    M_vec = []
    R_cm = np.zeros(3)
    I = np.zeros((3,3))
    I0 = 0

    #get mass of each atoms in the molecule
    #get coordinates of molecule
    #set cm coordinates
    for i in range(natom):
        coordinates[i,:] = np.array([molecule.x(i),molecule.y(i),molecule.z(i)])
        R_cm += molecule.mass(i)*coordinates[i,:]
        lst = [molecule.mass(i),molecule.mass(i),molecule.mass(i)]
        M_vec += lst
    
    M_vec = np.array(M_vec)
    R_cm = R_cm*3/np.sum(M_vec)
    cm_coordinates = coordinates - R_cm

    #Calculate inertia tensor
    for i in range(natom):
        I += molecule.mass(i)*np.outer(cm_coordinates[i,:],cm_coordinates[i,:])
        I0 += molecule.mass(i)*np.dot(cm_coordinates[i,:],cm_coordinates[i,:])
    I = I0*np.eye(3) - I

    Ieigval,X = np.linalg.eig(I) #get principle vectors.
    # note that {X[0,:],X[1,:],X[2,:]} forms the Eckart frame

    # Translational basis
    S1 = np.zeros((natom,3))
    S2 = np.zeros((natom,3))
    S3 = np.zeros((natom,3))

    # Rotational basis
    S4 = np.zeros((natom,3))
    S5 = np.zeros((natom,3))
    S6 = np.zeros((natom,3))

    for i in range(natom):
        #Eckart condition for translational motion
        S1[i,:] = X[:,0].T
        S2[i,:] = X[:,1].T
        S3[i,:] = X[:,2].T

        #Eckart condition for rotational motion
        S4[i,:] = np.cross(cm_coordinates[i,:],X[:,0].T)
        S5[i,:] = np.cross(cm_coordinates[i,:],X[:,1].T)
        S6[i,:] = np.cross(cm_coordinates[i,:],X[:,2].T)

    S1 = np.sqrt(M_vec) * S1.flatten()
    S2 = np.sqrt(M_vec) * S2.flatten()
    S3 = np.sqrt(M_vec) * S3.flatten()
    S4 = np.sqrt(M_vec) * S4.flatten()
    S5 = np.sqrt(M_vec) * S5.flatten()
    S6 = np.sqrt(M_vec) * S6.flatten()

    # To get orthonomal basis which extends to {S1,...S6}, 
    # orthonormalize {S1,..,S6,e1,...e3natom} using QR decomposition algorithm
    S = np.zeros((3*natom,3*natom+6))

    S[:,0] = S1.T
    S[:,1] = S2.T
    S[:,2] = S3.T
    S[:,3] = S4.T
    S[:,4] = S5.T
    S[:,5] = S6.T
    S[:,6:] = np.eye(3*natom) 

    D,tmp = np.linalg.qr(S) 
    #D: orthonormal basis extended by {S1,..,S6}
    #So, we can decompose R^3natom = R^ext + R^int using the basis D.

    H_card = np.array(hessian)
    H_weight = (np.outer(M_vec,M_vec))**(-1/2)*H_card
    H_ext_int = D.T @ H_weight @ D 

    #change the basis of hessian from the standard mass weight coordinates to
    #the orthonormal basis containing S1,...S6
    #Then H_ext_int = H_ext | 0
    #                 -------------
    #                   0   | H_int 

    H_int = H_ext_int[6:,6:]
    eig_normal,q_normal_reduced = np.linalg.eig(H_int)

    # Note that for psi4, the unit of weighted hessian is hartree/(bohr^2*amu)
    # So we must multiply conversion factor to get correct unit(J*kg**(-1)*m**(-2) =Hz^2) for the
    # eigen value of internal hessian eig_normal
    eig_normal_converted = eig_normal*unit_conversion_factor 
    
    #1/100 is the conversion factor for m**(-1) to cm**(-1)
    vib_freq = np.sqrt(eig_normal_converted)/(2*np.pi*speed_of_light)*1/100

    q_normal_full = np.zeros((3*natom,3*natom-6))
    q_normal_full[6:,:] = q_normal_reduced

    #                     D
    #internal coordinate --> mass weight cartesian coordinate 
    disp_normal = D.T @ q_normal_full 

    return vib_freq,disp_normal

    