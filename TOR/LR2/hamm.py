import numpy as np

def corr(Nel,d_element):
    output = np.arange(Nel) * d_element - ((Nel - 1) * d_element)/2
    return output

def hamming(coord):
    L_ant = coord[len(coord)-1] - coord[0]
    output = 0.54+0.46*np.cos(2*np.pi*coord/L_ant)
    return output

def convert(input,razr):
    norm = 2**razr
    output = np.round(input*norm)
    return output

def phase_dist(coord,lyambda,angle):
    k = 2*np.pi/lyambda
    output = k*coord*np.sin(angle)
    return output


if __name__ == '__main__':
    Nel = 16
    d_element = 8.7e-3
    wave_lenght = 8.575e-3
    razryadnost_ampl = 8
    razryadnost_phase = 12
    phi = -20*np.pi/180
    coordinats = corr(Nel,d_element)
    print("Координаты элементов антенны",coordinats)
    hamm_raw = hamming(coordinats)
    print ("Хамминг относительный:",hamm_raw)
    hamm_welldone = convert(hamm_raw,razryadnost_ampl)
    print ("Хамминг битовый:",hamm_welldone)
    phase_raw = phase_dist(coordinats,wave_lenght,phi)
    print ("Распределние фаз:",phase_raw)
    phase_odnoznach = phase_raw - np.floor(phase_raw / (2*np.pi))*2*np.pi
    print ("Распределение фаз в пределах:",phase_odnoznach)
    phase_welldone = convert(phase_odnoznach/(2*np.pi),razryadnost_phase)
    print ("Распределение фаз относительное:",phase_welldone)


