

#set the geologically meaningful rake categories based on the rake angle
def set_rakes(pontok):
    pontok.r.loc[pontok.r==0] = 1 #left_strike_slip
    pontok.r.loc[(pontok.r>0) & (pontok.r<20)] = 1 #left lateral strike-slip
    pontok.r.loc[(pontok.r>=20) & (pontok.r<70)] = 12 #reverse left-lateral oblique
    pontok.r.loc[(pontok.r>=70) & (pontok.r<110)] = 13 #reverse
    pontok.r.loc[(pontok.r>=110) & (pontok.r<160)] = 14 #reverse right-lateral oblique
    pontok.r.loc[(pontok.r>=160) & (pontok.r<=180)] = 1 #left lateral strike-slip
    pontok.r.loc[pontok.r==180] = 2 #right_strike_slip
    r_list_n = [0, -20, -70, -110, -160, -180]
    pontok.r.loc[((pontok.r>r_list_n[1]) & (pontok.r<=r_list_n[0]))] = 1 #left lateral strike-slip
    pontok.r.loc[((pontok.r>r_list_n[2]) & (pontok.r<=r_list_n[1]))] = 22 #normal left-lateral oblique
    pontok.r.loc[((pontok.r>r_list_n[3]) & (pontok.r<=r_list_n[2]))] = 23 #normal
    pontok.r.loc[((pontok.r>r_list_n[4]) & (pontok.r<=r_list_n[3]))] = 24 #normal right-lateral oblique
    pontok.r.loc[((pontok.r>=r_list_n[5]) & (pontok.r<=r_list_n[4]))] = 1 #left lateral strike-slip
    pontok.r.loc[((pontok.r>180-r_list_n[0]) & (pontok.r<=180-r_list_n[1]))] = 2 #right lateral strike-slip
    pontok.r.loc[((pontok.r>180-r_list_n[1]) & (pontok.r<=180-r_list_n[2]))] = 22 #normal left-lateral oblique
    pontok.r.loc[((pontok.r>180-r_list_n[2]) & (pontok.r<=180-r_list_n[3]))] = 23 #normal
    pontok.r.loc[((pontok.r>180-r_list_n[3]) & (pontok.r<=180-r_list_n[4]))] = 24 #normal right-lateral oblique
    pontok.r.loc[((pontok.r>=180-r_list_n[4]) & (pontok.r<=180-r_list_n[5]))] = 1 #left lateral strike-slip
    pontok.r.loc[(pontok.r==270) | (pontok.r==-90)] = 21 #normal dip-slip
    pontok.r.loc[pontok.r==90] = 11 #reverse-dip-slip
    return pontok