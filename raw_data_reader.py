#!/usr/bin/env python3

import argparse
import os

import cdflib
import pandas as pd
import numpy as np


# this is how you turn counts to EFLUX to VDF
def createVDF(cdfData):

    '''
    Takes a CDF file object and creates an velocity distribution function in Vx, Vy, Vz from EFFLUX data.

    This function is partially built on code found within tutorials found online for reading PSP data.

    https://github.com/jlverniero/PSP_Data_Analysis_Tutorials/blob/main/README.md
    '''

    # 0. Define useful constants and file information
    qFlagKey = np.array([
            'Bit0: Counter Overflow',
            'Bit1: Survey Snapshot ON',
            'Bit2: Alternate Energy Table',
            'Bit3: Spoiler Test',
            'Bit4: Attenuator Engaged',
            'Bit5: Highest Archive Rate',
            'Bit6: No Targeted Sweep',
            'Bit7: SPAN-Ion New Mass Table',
            'Bit8: Over-deflection',
            'Bit9: Archive Snapshot ON',
            'Bit10: Bad energy table',
            'Bit11: MCP Test',
            'Bit12: Survey available',
            'Bit13: Archive available',
            'Bit14: RESERVED',
            'Bit15: RESERVED'])

    mass_p = 0.01043970 # proton mass
                        # [eV/(km/s)^2]
                        # (938.272 * 10**6) / (299792.458**2)

    # 1. Read in useful information from file to np.ndarrays
    numEntries = len(cdfData['TIME'])

    eFlux      = cdfData['EFlux'].reshape((numEntries, 8, 32, 8)) # [eV/cm2-s-ster-eV]
    counts     = cdfData['DATA'].reshape((numEntries, 8, 32, 8)) # [#]
    time_accum = cdfData['TIME_ACCUM']

    gFactor = (counts / time_accum[:, np.newaxis, np.newaxis, np.newaxis]) / eFlux # counts/time_accum/eflux -- Counts per second / eflux

    # coords
    # organized as an 8 phi-direction, 32 energy bins, and 8 theta-direction bins
    energy = cdfData['ENERGY'].reshape((numEntries, 8, 32, 8)) # [eV]
    theta  = cdfData['THETA'].reshape((numEntries, 8, 32, 8))  # [degree]
    phi    = cdfData['PHI'].reshape((numEntries, 8, 32, 8))    # [degree]

    # Calculate the one count level in vdf.
    # we assume that when we can not calculate the gFactor due to restrictions in
    # the access to public information, a reasonable assumption is that
    # the gFactor is roughly similar to the mean.
    # This is currently an unconstrained uncertainty.

    fillVals = np.nanmean(gFactor, axis = (1, 2, 3)) # np.nanmean(gfactor_day, axis = (1, 2, 3))

    for day_i in np.arange(0, np.shape(gFactor)[0]):
        # this is slow; future implementations should focus on speed ups through this
        # avenue
        gFactor[day_i, :, :, :] = np.nan_to_num(gFactor[day_i, :, :, :], nan = fillVals[day_i], copy = False)

    eFlux_one_count = (1 / time_accum[:, np.newaxis, np.newaxis, np.newaxis]) / gFactor

    # handle quality flags by returns an (nTime, 16) array
    # of all active quality flags or nan (if unactive)
    qualityFlags    = np.unpackbits(cdfData['QUALITY_FLAG'].view(np.uint8), bitorder = 'little')
    qFlags          = qualityFlags.reshape(len(cdfData['QUALITY_FLAG']), 16).astype(np.int32)

    reshaped_qFlags = np.tile(qFlagKey, len(cdfData['TIME'])).reshape(len(cdfData['TIME']), 16)
    reshaped_qFlags[qFlags == 0] = np.nan

    # 2. Calculate velocity distriubtion function
    # note; a derivation of the unit conversion from number flux to VDF
    # is provided in G. Hanley's 2023 PhD Thesis

    numberFlux = eFlux / energy # [#/cm2-s-ster-eV]

    numberFlux_one_count = eFlux_one_count / energy

    vdf_temp   = numberFlux*(mass_p**2)/((2 * 10**5)*energy) # [#/cm^3 (km/s)^3]

    vdf_m      = vdf_temp * (100**3) # [# / m^3 (km/s)^3]

    vdf_dm     = vdf_temp * (10**3) # [# / dm^3 (km/s)^3], 1 decimeter = 10 cm

    vdf        = vdf_m

    # one count

    vdf_temp_one_count   = numberFlux_one_count*(mass_p**2)/((2 * 10**5)*energy) # [#/cm^3 (km/s)^3]

    vdf_m_one_count      = vdf_temp_one_count * (100**3) # [# / m^3 (km/s)^3]

    vdf_dm_one_count     = vdf_temp_one_count * (10**3) # [# / dm^3 (km/s)^3], 1 decimeter = 10 cm

    vdf_one_count        = vdf_m_one_count

    # Convert to velocity units in each energy channel
    vel = np.sqrt(2*energy/mass_p) # [km/s]

    # rotate from spherical to Cartesian
    vx = vel * np.cos(np.radians(phi)) * np.cos(np.radians(theta))
    vy = vel * np.sin(np.radians(phi)) * np.cos(np.radians(theta))
    vz = vel *                           np.sin(np.radians(theta))

    return(reshaped_qFlags, vdf, vdf_one_count, counts, vx, vy, vz)


def calcRMatrix(L3Data):
    '''
    Estimates the rotation matrix from L3 data from instrument frame to a magnetic field coordinate frame.
    '''

    B = L3Data['MAGF_INST']

    B_mag = np.linalg.norm(B, axis = 1, keepdims = True)

    j_prime = B / B_mag

    # find the least directional axes
    # set this new axis to be the temp cross direction

    indexMin = (j_prime == np.min(j_prime, 1, keepdims = True))

    j = np.zeros(np.shape(j_prime))

    j[indexMin] = 1

    i_prime_unnormed = np.cross(j_prime, j)

    i_prime = i_prime_unnormed / np.linalg.norm(i_prime_unnormed, axis = 1, keepdims = True)

    k_prime_unnormed = np.cross(i_prime, j_prime)

    k_prime = k_prime_unnormed / np.linalg.norm(k_prime_unnormed, axis = 1, keepdims = True)

    A = np.stack([i_prime, j_prime, k_prime], 1)

    return(A)


def main():
    parser = argparse.ArgumentParser(description="Export one PSP VDF snapshot to CSV (no calculation changes).")
    parser.add_argument("--l2", required=True, help="Path to L2 CDF file")
    parser.add_argument("--l3", required=True, help="Path to L3 CDF file")
    parser.add_argument("--outdir", required=True, help="Directory to write CSV output")
    parser.add_argument("--i", type=int, default=2, help="Eligible-filter index i (same as your notebook)")
    parser.add_argument("--rmax", type=float, default=40.0, help="R_max in solar radii (same default as your notebook)")
    args = parser.parse_args()

    R_sun = 695700000 # [meters]
    R_max = args.rmax # [Solar radii]

    L2Data = cdflib.CDF(args.l2)
    L3Data = cdflib.CDF(args.l3)

    qFlags_day, vdf_day, vdf_one_count_day, counts_day, vx_day, vy_day, vz_day = createVDF(L2Data)

    A_day = calcRMatrix(L3Data)

    times = pd.to_datetime(L2Data['TIME'], unit = 's')

    eligibleFilter = (L3Data['SUN_DIST'] / (R_sun/10**3)) < R_max

    # look at ONE vdf and not a DAY's VDF values, let's say the first one
    # in an eligible filter range?

    i = args.i

    r_dist = L3Data['SUN_DIST'][eligibleFilter][i] / (R_sun/10**3)

    qFlags, vdf, vdf_one_count, vx, vy, vz, counts = qFlags_day[eligibleFilter][i], vdf_day[eligibleFilter][i], vdf_one_count_day[eligibleFilter][i], vx_day[eligibleFilter][i], vy_day[eligibleFilter][i], vz_day[eligibleFilter][i], counts_day[eligibleFilter][i]

    counts[:, [0], :] = 0
    vdf[:,    [0], :] = 0

    vdfDate = pd.to_datetime(L2Data['TIME'][eligibleFilter][i], unit = 's')

    A = A_day[i]

    v_perp_0, v_para, v_perp_1 = A @ np.array([vx.ravel(), vy.ravel(), vz.ravel()])

    v_perp_0 = np.reshape(v_perp_0, np.shape(vx))

    v_para   = np.reshape(v_para, np.shape(vx))

    v_perp_1 = np.reshape(v_perp_1, np.shape(vx))

    df = pd.DataFrame({
        "v_para":   v_para.ravel(),
        "v_perp_0": v_perp_0.ravel(),
        "v_perp_1": v_perp_1.ravel(),
        "f(v)":     vdf.ravel(),
        "counts":   counts.ravel()
    })

    timestamp = pd.to_datetime(L2Data['TIME'][eligibleFilter][i], unit='s')
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    filename = f"vdf_{timestamp_str}.csv"
    outpath = os.path.join(args.outdir, filename)

    df.to_csv(outpath, index=False)

    print(f"Wrote CSV to: {outpath}")


if __name__ == "__main__":
    main()
    