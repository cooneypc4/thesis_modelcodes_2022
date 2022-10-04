import h5py 


def loadconns_roll():
    f = h5py.File("gendata/data_roll_attemptpmnexcit-3-lccwlarva3-small-zeros.h5","r")
    Jpm = ((f["Jpm"][:]).T)
    Jpp = ((f["Jpp"][:]).T)

    pnames = f["p"][:]
    mnames = f["m"][:]
    types = f["nt"][:]
    mnorder = f["mnorder"][:]
    f.close()

    return Jpm,Jpp,pnames,mnames,types,mnorder

