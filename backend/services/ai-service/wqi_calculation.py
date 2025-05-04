def calculate_PH_value(ph):
    if ph < 5.5:
        return 1
    elif 5.5 <= ph < 6.5:
        return 99 * ph - 543.5
    elif 6.5 <= ph < 8.5:
        return 100
    elif 8.5 <= ph < 9.5:
        return -99 * ph + 941.5
    else:  # ph >= 9.5
        return 1
#Convert the Do dan vlue
def calculate_EC_value(value):
    if value <= 1500:
        return 100
    elif 1500 < value < 4500:
        return (-0.033*value + 149.5)
    else:  # ph >= 4500
        return 1
#Convert the Dissolved oxygen (DO) value
def calculate_qDO(DO):
    if DO <= 3:
        return 1
    elif 3 < DO < 5:
        return 49.5 * DO - 147.5
    elif 5 <= DO < 7:
        return 100
    elif 7 <= DO < 11:
        return -24.75 * DO + 273.25
    else:  # DO >= 11
        return 1
#Convert the TSS value
def calculate_qTSS(TSS):
    if TSS <= 50:
        return 100
    elif 50 < TSS < 150:
        return -0.99 * TSS + 149.5
    else:
        return 1
#Convert the COD value
def calculate_qCOD(COD):
    if COD <= 10:
        return 100
    elif 10 < COD < 20:
        return -9.9 * COD + 199
    else:  # COD >= 20
        return 1
#Convert the N-NH4 value
def calculate_qNNH4(vl):
    if vl <= 0.3:
        return 100
    elif 0.3 < vl < 1.7:
        return -70.71 * vl + 121.21
    else:  # N-NH4 >= 1.7
        return 1
#Convert the N-NO2 value
def calculate_qNNO2(vl):
    if vl <= 0.1:
        return 100
    elif 0.1 < vl < 1:
        return -111.1 * vl + 111
    else:  # N-NO2 >= 1
        return 1
#Convert the P-PO4 value
def calculate_qPPO4(vl):
    if vl <= 0.1:
        return 100
    elif 0.1 < vl < 0.5:
        return -247.5 * vl + 124.75
    else:  # P-PO4 >= 0.5
        return 1
#Convert the Aeromonas value
def calculate_qAeromonas(vl):
    if vl <= 1000:
        return 100
    elif 1000 < vl < 3000:
        return -0.0495 * vl + 149.5
    else:  # Aeromonas >= 3000
        return 1
def safe_power(x, power, decimals=8):
    if x < 0:
        return None
    else:
        return round(x ** power, decimals)
def calculate_WQI(ph, DO, EC, N_NO2, N_NH4, P_PO4, TSS, COD, Aeromonas):
    # Step 1: Convert each input to a quality value
    q_ph = safe_power(calculate_PH_value(ph), 0.11)
    q_EC = safe_power(calculate_EC_value(EC), 0.06)
    q_DO = safe_power(calculate_qDO(DO), 0.1)
    q_TSS = safe_power(calculate_qTSS(TSS), 0.13)
    q_COD = safe_power(calculate_qCOD(COD), 0.1)
    q_N_NH4 = safe_power(calculate_qNNH4(N_NH4), 0.13)
    q_N_NO2 = safe_power(calculate_qNNO2(N_NO2), 0.1)
    q_P_PO4 = safe_power(calculate_qPPO4(P_PO4), 0.12)
    q_Aeromonas = safe_power(calculate_qAeromonas(Aeromonas), 0.15)

    # Step 2: Multiply the quality values to get the WQI
    WQI = round(q_ph * q_EC * q_DO * q_TSS * q_COD * q_N_NH4 * q_N_NO2 * q_P_PO4 * q_Aeromonas, 4)
    return WQI