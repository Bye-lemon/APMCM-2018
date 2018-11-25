# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft

from GrayModel import Gray_model


def get_raw_data():
    '''
        Get Total Demand Data From APMCM Raw Data
    '''
    total = dict()
    for (root, dirs, files) in os.walk("data"):
        for file in files:
            path = os.path.join(root, file)
            df = pd.read_excel(path, index_col=1, names=["S/N", "Total", "Junior", "Senior", "Tech", "Junior College", "Bach", "Master", "Doctor", "MBA", "Unlimited"])
            df = df.iloc[2:-2]
            ser = df["Total"]
            ser.rename(columns={"Total": path})
            ser = ser.dropna(axis=0, how='all')
            total.update({path: ser})
    data = pd.DataFrame(total)
    return data

def write_excel(data, path):
    '''
        Write Frame Data To An Excel Sheet
    
        data: Pandas DataFrame
        path: String
    '''
    data.to_excel(path)

def nan_to_zero(data):
    '''
        Replace NaN With Zero
        
        data: Pandas DataFrame
    '''
    return data.replace(np.nan, 0)

def test_get_raw_data():
    data = get_raw_data()
    data = nan_to_zero(data)
    write_excel(data, 'total.xlsx')
    for i in range(len(data)-1):
        pic = data.iloc[i].plot(title=data.index[i])
        fig = pic.get_figure()
        fig.savefig('image/' + data.index[i].replace('/', ' ') + '.png')
        plt.show()
    else:
        pic = data.iloc[i+1].plot(title='NAN')
        fig = pic.get_figure()
        fig.savefig('image/NAN.png')
        plt.show()
        
def build_data_monthly(col_name):
    total = dict()
    for item in range(36):
        df = pd.read_excel("Monthly.xlsx", sheet_name=item, index_col=0, names=["Rate", "Require", "Offer", "Unlimited", "Above", "Below", "Bach"])
        df = df.iloc[2:]
        ser = df[col_name]
        ser.rename(columns={col_name: item})
        total.update({item: ser})
    data = pd.DataFrame(total)
    data = nan_to_zero(data)
    write_excel(data, 'mid_data/Monthly/' + col_name + '.xlsx')

def build_delta_monthly():
    total = {}
    for item in range(36):
        df = pd.read_excel("Monthly.xlsx", sheet_name=item, index_col=0, names=["Rate", "Require", "Offer", "Unlimited", "Above", "Below", "Bach"])
        df = df.iloc[2:]
        ser1 = df["Require"]
        ser2 = df["Offer"]
        ser = ser2 - ser1
        ser.rename(columns={'Delta': item})
        total.update({item: ser})
    data = pd.DataFrame(total)
    data = nan_to_zero(data)
    write_excel(data, 'mid_data/Monthly/Delta.xlsx')

def build_delta_yearly():
    total = {}
    for item in range(1, 4):
        df = pd.read_excel("Yearly.xlsx", sheet_name=item, index_col=0, names=["Below", "Bach", "Above", "Unlimited", "Offer", "Require", "Rate", "N_Below", "N_Bach", "N_Above", "N_Unlimited"])
        ser1 = df["Require"]
        ser2 = df["Offer"]
        ser = ser2 - ser1
        ser.rename(columns={'Delta': item})
        total.update({item: ser})
    data = pd.DataFrame(total)
    data = nan_to_zero(data)
    write_excel(data, 'mid_data/Yearly/Delta.xlsx')
    
def build_data_yearly(col_name):
    total = dict()
    for item in range(1, 4):
        df = pd.read_excel("Yearly.xlsx", sheet_name=item, index_col=0, names=["Below", "Bach", "Above", "Unlimited", "Offer", "Require", "Rate", "N_Below", "N_Bach", "N_Above", "N_Unlimited"])
        ser = df[col_name]
        ser.rename(columns={col_name: item})
        total.update({item: ser})
    data = pd.DataFrame(total)
    data = nan_to_zero(data)
    write_excel(data, 'mid_data/Yearly/' + col_name + '.xlsx')


def generate_data_monthly():
    print("Monthly Data Generating ...")
    columns=["Rate", "Require", "Offer", "Unlimited", "Above", "Below", "Bach"]
    for item in columns:
        build_data_monthly(item)
        print(item + " Table Complete")
    build_delta_monthly()
    print("Delta Table Complete")
    
def generate_data_yearly():
    print("Yearly Data Generating ...")
    columns=["Below", "Bach", "Above", "Unlimited", "Offer", "Require", "Rate"]
    for item in columns:
        build_data_yearly(item)
        print(item + " Table Complete")
    build_delta_yearly()
    print("Delta Table Complete")
    
def generate_data():
    generate_data_monthly()
    generate_data_yearly()
        
def linear_fit(ser):
    '''
        Generate Fit Line Of A Serial And Plot It.
        
        ser: Pandas Serial
    '''
    list = ser.tolist()
    x = [item for item in range(len(list))]
    yy = np.poly1d(np.polyfit(x, list, 1))
    return yy
    

def fourier(list, name, save_flag=False):
    y = np.array(list)
    x = np.arange(0, 1.0, 1.0/len(y))
    yy = fft(y)
    yreal = yy.real
    yimag = yy.imag
    xlist = np.arange(0, 1.0, 1/len(y))
    ylist = []
    for x in xlist:
        yyy = 0
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)-0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)+0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        ylist.append(yyy/2)
    ymean = y.sum()/len(y)
    #plt.plot(xlist, ylist)
    ylistmean = sum(ylist)/len(ylist)
    乘系数y = []
    for item in ylist:
        乘系数y.append(item*float(ymean/ylistmean))
    delta = [(y[i]-乘系数y[i]) for i in range(len(y))]
    分母 = sum([(y[i]-ymean)**2 for i in range(len(y))])
    分子 = sum([delta[i]**2 for i in range(len(y))])
    if save_flag:
        fig = plt.plot(xlist, y)
        fig = plt.plot(xlist, 乘系数y)
        plt.savefig("image/Fourier/"+name.replace('/', ' ')+".png")
    plt.show()
    return 1-分子/分母
    
def test_fourier():
    list=[150,140,113,92,34,459,534,191,193,130,166,176,137,124,87,75,11,787,398,169,205,140,133,188,163,174,98,66,31,177,666,160,208,137,133,151]
    fourier(list)
    
def judge_季度型还是非季度性(month):
    request = pd.read_excel("mid_data/Monthly/Require.xlsx")
    offer = pd.read_excel("mid_data/Monthly/Offer.xlsx")
    ser_r = request.iloc[month]
    ser_o = offer.iloc[month]
    return (fourier(ser_r.tolist(), request.index[month]) > 0.11) or (fourier(ser_o.tolist(), offer.index[month]) > 0.11)

def calc_季度性与非季度性的判据矩阵():
    dic = {}
    df = pd.read_excel("mid_data/Monthly/Offer.xlsx")
    for i in range(50):
        dic.update({df.index[i]: judge_季度型还是非季度性(i)})
    判据矩阵 = pd.DataFrame(pd.Series(dic, name="Bool"))
    return 判据矩阵

def calc_无季度权重():
    columns = ["Above", "Below", "Bach"]
    ser_list = []
    df = pd.read_excel("mid_data/Monthly/Above.xlsx")
    dic = {}
    for i in range(50):
        dic.update({df.index[i]: 1})
    ser_list.append(pd.Series(dic, name='Delta'))
    for item in columns:
        df = pd.read_excel("mid_data/Monthly/" + item + ".xlsx")
        dic = {}
        for i in range(50):
            ser = df.iloc[i]
            list = ser.tolist()
            x = [item for item in range(len(list))]
            yy = np.poly1d(np.polyfit(x, list, 1))
            yyy = yy(x)
            sum = 0
            for i in range(len(x)):
                sum += (list[i]-yyy[i])**2/yyy[i]**2
            cost = (1-sum/len(x)) if ((1-sum/len(x)) >= 0) else 0
            dic.update({ser.name: cost})
        ser = pd.Series(dic, name=item)
        ser_list.append(ser)
    S_Martix = pd.DataFrame(ser_list).T
    return S_Martix

def calc_有季度权重():
    columns = ["Above", "Below", "Bach"]
    ser_list = []
    df = pd.read_excel("mid_data/Yearly/Above.xlsx", index_col=0)
    dic = {}
    for i in range(50):
        dic.update({df.index[i]: 1})
    ser_list.append(pd.Series(dic, name='Delta'))
    for item in columns:
        df = pd.read_excel("mid_data/Yearly/" + item + ".xlsx", index_col=0)
        dic = {}
        for i in range(50):
            ser = df.iloc[i]
            list = ser.tolist()
            x = [item for item in range(len(list))]
            yy = np.poly1d(np.polyfit(x, list, 1))
            yyy = yy(x)
            sum = 0
            for i in range(len(x)):
                sum += (list[i]-yyy[i])**2/yyy[i]**2
            cost = (1-sum/len(x)) if ((1-sum/len(x)) >= 0) else 0
            dic.update({ser.name: cost})
        ser = pd.Series(dic, name=item)
        ser_list.append(ser)
    S_Martix = pd.DataFrame(ser_list).T
    return S_Martix

def calc_无季度数量型数值(month):
    columns = ["Above", "Below", "Bach"]
    total = {}
    df = pd.read_excel("mid_data/Monthly/Delta.xlsx")
    ser = df.loc[:,month]
    total.update({'Delta': ser})
    df = pd.read_excel("mid_data/Monthly/Offer.xlsx")
    offer = df.loc[:,month]
    for item in columns:
        df = pd.read_excel("mid_data/Monthly/"+item+".xlsx")
        ser = offer.multiply(df.loc[:,month])
        total.update({item: ser})
    D_Martix = pd.DataFrame(total)
    return D_Martix

def calc_有季度数量型数值(year):
    columns = ["Above", "Below", "Bach"]
    total = {}
    df = pd.read_excel("mid_data/Yearly/Delta.xlsx", index_col=0)
    ser = df.loc[:,year]
    total.update({'Delta': ser})
    df = pd.read_excel("mid_data/Yearly/Offer.xlsx", index_col=0)
    offer = df.loc[:,year]
    for item in columns:
        df = pd.read_excel("mid_data/Yearly/"+item+".xlsx", index_col=0)
        ser = offer.multiply(df.loc[:,year])
        total.update({item: ser})
    D_Martix = pd.DataFrame(total)
    return D_Martix

def calc_无季度数量型矩阵(month):
    SM = calc_无季度权重()
    DM = calc_无季度数量型数值(month)
    MM = SM.multiply(DM)
    MM_l = MM.iloc[:, :4]
    return MM_l

def calc_有季度数量型矩阵(year):
    SM = calc_有季度权重()
    DM = calc_有季度数量型数值(year)
    MM = SM.multiply(DM)
    MM_l = MM.iloc[:, :4]
    return MM_l

def calc_无季度比例型数值(month):
    columns = ["Rate", "Above", "Below", "Bach"]
    total = {}
    for item in columns:
        df = pd.read_excel("mid_data/Monthly/"+item+".xlsx")
        ser = df.loc[:,month]
        total.update({item: ser})
    D_Martix = pd.DataFrame(total)
    return D_Martix

def calc_有季度比例型数值(year):
    columns = ["Rate", "Above", "Below", "Bach"]
    total = {}
    for item in columns:
        df = pd.read_excel("mid_data/Yearly/"+item+".xlsx", index_col=0)
        ser = df.loc[:,year]
        total.update({item: ser})
    D_Martix = pd.DataFrame(total)
    return D_Martix

def calc_无季度比例型矩阵(month):
    SM = calc_无季度权重()
    SM.rename(columns={"Delta": "Rate"}, inplace=True)
    DM = calc_无季度比例型数值(month)
    MM = SM.multiply(DM)
    return MM
    
def calc_有季度比例型矩阵(year):
    SM = calc_有季度权重()
    SM.rename(columns={"Delta": "Rate"}, inplace=True)
    DM = calc_有季度比例型数值(year)
    MM = SM.multiply(DM)
    return MM

def calc_真正的有季度数量型矩阵(year):
    判据矩阵 = calc_季度性与非季度性的判据矩阵()
    有季度数量型矩阵 = calc_有季度数量型矩阵(year+1)
    数量型矩阵 = 有季度数量型矩阵
    droplist = []
    for i in range(50):
        if 判据矩阵.iloc[i].bool() == False:
            droplist.append(数量型矩阵.index[i])
    return 数量型矩阵.drop(droplist)

def calc_真正的无季度数量型矩阵(month):
    判据矩阵 = calc_季度性与非季度性的判据矩阵()
    无季度数量型矩阵 = calc_无季度数量型矩阵(month)
    数量型矩阵 = 无季度数量型矩阵
    droplist = []
    for i in range(50):
        if 判据矩阵.iloc[i].bool() == True:
            droplist.append(数量型矩阵.index[i])
    return 数量型矩阵.drop(droplist)

def calc_真正的有季度比例型矩阵(year):
    判据矩阵 = calc_季度性与非季度性的判据矩阵()
    有季度比例型矩阵 = calc_有季度比例型矩阵(year+1)
    比例型矩阵 = 有季度比例型矩阵
    droplist = []
    for i in range(50):
        if 判据矩阵.iloc[i].bool() == False:
            droplist.append(比例型矩阵.index[i])
    return 比例型矩阵.drop(droplist)

def calc_真正的无季度比例型矩阵(month):
    判据矩阵 = calc_季度性与非季度性的判据矩阵()
    无季度比例型矩阵 = calc_无季度比例型矩阵(month)
    比例型矩阵 = 无季度比例型矩阵
    droplist = []
    for i in range(50):
        if 判据矩阵.iloc[i].bool() == True:
            droplist.append(比例型矩阵.index[i])
    return 比例型矩阵.drop(droplist)

def 矩阵归一化(df):
    # Rate: 成本型
    min_r = df.iloc[:, 0].min()
    max_r = df.iloc[:, 0].max()
    df.iloc[:, 0] = (max_r-df.iloc[:, 0])/(max_r-min_r)
    # Above: 效益型
    max_a = df.iloc[:, 1].max()
    df.iloc[:, 1] = df.iloc[:, 1]/max_a
    # Below：成本型
    min_bl = df.iloc[:,2].min()
    max_bl = df.iloc[:, 2].max()
    df.iloc[:, 2] = (max_bl-df.iloc[:, 2])/(max_bl-min_bl)
    # Bach：效益型
    max_b = df.iloc[:, 3].max()
    df.iloc[:, 3] = df.iloc[:, 3]/max_b
    return df

def calc_得分矩阵(df):
    weight = [0.2428, 0.3755, 0.2172, 0.1645]
    df = df*weight
    ser_list = []
    for i in range(len(df.index)):
        ser = pd.Series(df.iloc[i].sum(), name=df.index[i])
        ser_list.append(ser)
    score = pd.DataFrame(ser_list)
    return score

def calc_有季度平均得分矩阵(year):
    bl = calc_真正的有季度比例型矩阵(year)
    sl = calc_真正的有季度数量型矩阵(year)
    return calc_得分矩阵(矩阵归一化(sl)), calc_得分矩阵(矩阵归一化(bl)), (calc_得分矩阵(矩阵归一化(sl)) + calc_得分矩阵(矩阵归一化(bl)))/2

def calc_无季度平均得分矩阵(month):
    bl = calc_真正的无季度比例型矩阵(month)
    sl = calc_真正的无季度数量型矩阵(month)
    return calc_得分矩阵(矩阵归一化(sl)), calc_得分矩阵(矩阵归一化(bl)), (calc_得分矩阵(矩阵归一化(sl)) + calc_得分矩阵(矩阵归一化(bl)))/2

def calc_总的有季度得分():
    ava = dict()
    sl = dict()
    bl = dict()
    for item in range(3):
        slf, blf, df = calc_有季度平均得分矩阵(item)
        ser = df.iloc[:, 0]
        ava.update({item: ser})
        ser = slf.iloc[:, 0]
        sl.update({item: ser})
        ser = blf.iloc[:, 0]
        bl.update({item: ser})
    ava = pd.DataFrame(ava).mean(1)
    sl = pd.DataFrame(sl).mean(1)
    bl = pd.DataFrame(bl).mean(1)
    data = pd.DataFrame({"Number": sl, "Scale": bl, "Average":ava})
    write_excel(data, 'mid_data/Yearly/Score.xlsx')


def calc_总的无季度得分():
    ava = dict()
    sl = dict()
    bl = dict()
    for item in range(36):
        slf, blf, df = calc_无季度平均得分矩阵(item)
        ser = df.iloc[:, 0]
        ava.update({item: ser})
        ser = slf.iloc[:, 0]
        sl.update({item: ser})
        ser = blf.iloc[:, 0]
        bl.update({item: ser})
    ava = pd.DataFrame(ava).mean(1)
    sl = pd.DataFrame(sl).mean(1)
    bl = pd.DataFrame(bl).mean(1)
    data = pd.DataFrame({"Number": sl, "Scale": bl, "Average":ava})
    write_excel(data, 'mid_data/Monthly/Score.xlsx')

def get_fourier_func(list):
    y = np.array(list)
    x = np.arange(0, 1.0, 1.0/len(y))
    yy = fft(y)
    yreal = yy.real
    yimag = yy.imag
    xlist = np.arange(0, 1.0, 1/len(y))
    ylist = []
    for x in xlist:
        yyy = 0
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)-0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)+0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        ylist.append(yyy/2)
    ymean = y.sum()/len(y)
    ylistmean = sum(ylist)/len(ylist)
    乘系数y = []
    for item in ylist:
        乘系数y.append(item*float(ymean/ylistmean))
    delta = [(y[i]-乘系数y[i]) for i in range(len(y))]
    delta_ser = pd.Series(delta)
    func = linear_fit(delta_ser)
    xxxx = [i for i in range(36, 72)]
    delta_pre = func(xxxx)
    x_pre_list = np.arange(1.0, 2.0, 1/len(y))
    y_pre_list = []
    for x in x_pre_list:
        yyy = 0
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)-0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        for item in range(0, len(yreal), 3):
            yyy += np.sqrt(yreal[item]*yreal[item]+yimag[item]*yimag[item])/len(y)*np.cos(2*np.pi*item*((x)+0.2)+np.arctan(yimag[item]/yreal[item]))
        yyy -= 0.5*np.sqrt(yreal[0]*yreal[0]+yimag[0]*yimag[0])/len(y)*np.cos(2*np.pi*0*(x)+np.arctan(yimag[0]/yreal[0]))
        y_pre_list.append(yyy/2*float(ymean/ylistmean))
    result = []
    for i in range(len(list)):
        res = y_pre_list[i]+delta_pre[i]
        ydropng = 0 if res < 0 else res
        result.append(ydropng)
    return result
    

def pre_data_build(col_name):
    data = pd.read_excel("mid_data/Monthly/"+col_name+".xlsx")
    ref = pd.read_excel("mid_data/Monthly/Require.xlsx")
    x = [i for i in range(36, 72)]
    ser_list = []
    for i in range(50):
        ser = data.iloc[i]
        ref_ser = ref.iloc[i]
        func = linear_fit(ser*ref_ser)
        y = func(x)
        ydropng = [(1 if item < 0 else item) for item in y]
        ser = pd.Series(ydropng, name=data.index[i])
        ser_list.append(ser)
    df = pd.DataFrame(ser_list)
    write_excel(df, "pre_data/"+col_name+"_pre.xlsx")
    
def pre_data_ro_build(col_name):
    data = pd.read_excel("mid_data/Monthly/"+col_name+".xlsx")
    judge = calc_季度性与非季度性的判据矩阵()
    ser_list = []
    for i in range(50):
        if judge.iloc[i].bool() == False:
            ser = data.iloc[i]
            gray = Gray_model()
            gray.fit(ser)
            result = gray.predict(36)
            ydropng = [(1 if item < 0 else item) for item in result]
            ser = pd.Series(ydropng, name=data.index[i])
            ser_list.append(ser)
        else:
            list = get_fourier_func(data.iloc[i].tolist())
            ser = pd.Series(list, name=data.index[i])
            ser_list.append(ser)
    df = pd.DataFrame(ser_list)
    write_excel(df, "pre_data/"+col_name+"_pre.xlsx")
    
def pre_data_generate():
    print("Monthly Data Predict ...")
    columns = ["Above", "Below", "Bach"]
    for item in columns:
        pre_data_build(item)
        print(item+" Table Complete")
    columns = ["Require", "Offer"]
    for item in columns:
        pre_data_ro_build(item)
        print(item+" Table Complete")
    print("Monthly Delta Generating ...")
    require = pd.read_excel("pre_data/Require_pre.xlsx")
    offer = pd.read_excel("pre_data/Offer_pre.xlsx")
    delta = offer - require
    write_excel(delta, "pre_data/Delta_pre.xlsx")
    

def calc_第二题数量型数值(month):
    columns = ["Above", "Below", "Bach"]
    total = {}
    df = pd.read_excel("pre_data/Delta_pre.xlsx", index_col=0)
    ser = df.loc[:,month]
    total.update({'Delta': ser})
    for item in columns:
        df = pd.read_excel("pre_data/"+item+"_pre.xlsx", index_col=0)
        ser = df.loc[:,month]
        total.update({item: ser})
    D_Martix = pd.DataFrame(total)
    return D_Martix

def calc_第二题数量型矩阵(month):
    SM = calc_无季度权重()
    DM = calc_第二题数量型数值(month)
    MM = SM.multiply(DM)
    return MM

def calc_第二题比例型数值(month):
    columns = ["Above", "Below", "Bach"]
    require = pd.read_excel("pre_data/Require_pre.xlsx", index_col=0)
    total = {}
    df = pd.read_excel("pre_data/Offer_pre.xlsx", index_col=0)
    ser = df.loc[:,month]
    ser_req = require.loc[:, month]
    total.update({"Delta": ser/ser_req})
    for item in columns:
        df = pd.read_excel("pre_data/"+item+"_pre.xlsx", index_col=0)
        ser = df.loc[:,month]
        ser_req = require.loc[:, month].sum()
        total.update({item: ser/ser_req})
    D_Martix = pd.DataFrame(total)
    D_Martix.replace(np.nan, 0, inplace=True)
    D_Martix.replace(np.inf, 0, inplace=True)
    return D_Martix

def calc_第二题比例型矩阵(month):
    SM = calc_无季度权重()
    DM = calc_第二题比例型数值(month)
    MM = SM.multiply(DM)
    return MM

def calc_第二题平均得分矩阵(month):
    bl = calc_第二题比例型矩阵(month)
    sl = calc_第二题数量型矩阵(month)
    return calc_得分矩阵(矩阵归一化(sl)), calc_得分矩阵(矩阵归一化(bl))


def calc_第二题得分():
    sl = dict()
    bl = dict()
    for item in range(36):
        slf, blf = calc_第二题平均得分矩阵(item)
        ser = slf.iloc[:, 0]
        sl.update({item: ser})
        ser = blf.iloc[:, 0]
        bl.update({item: ser})
    sl = pd.DataFrame(sl)
    bl = pd.DataFrame(bl)
    write_excel(sl, 'pre_data/Score_Number.xlsx')
    write_excel(bl, 'pre_data/Score_Scale.xlsx')
    
def 比例型得分打表():
    bl = dict()
    for item in range(3):
        slf, blf, df = calc_有季度平均得分矩阵(item)
        ser = blf.iloc[:, 0]
        bl.update({item: ser})
    bl = pd.DataFrame(bl)
    write_excel(bl, 'YearlyBili.xlsx')
    bl = dict()
    for item in range(36):
        slf, blf, df = calc_无季度平均得分矩阵(item)
        ser = blf.iloc[:, 0]
        bl.update({item: ser})
    bl = pd.DataFrame(bl)
    write_excel(bl, 'MonthlyBili.xlsx')
    
def calc_linear_para():
    df = pd.read_excel("out_data/第二题.xlsx", index_col=0)
    df = df.drop(["类"], axis=1)
    adict = {}
    bdict = {}
    for i in range(50):
        a = float(str(linear_fit(df.iloc[i])).strip().split(' ')[0])
        b = float(str(linear_fit(df.iloc[i])).strip().split(' ')[3])
        adict.update({df.index[i]: a})
        bdict.update({df.index[i]: b})
    aser = pd.Series(adict, name="a")
    bser = pd.Series(bdict, name="b")
    data = pd.DataFrame([aser, bser]).T
    write_excel(data, 'out_data/LinearFit_Second.xlsx')
    df = pd.read_excel("out_data/第三题.xlsx", index_col=0).replace(np.nan, 0)
    adict = {}
    bdict = {}
    for i in range(10):
        a = float(str(linear_fit(df.iloc[i])).strip().split(' ')[0])
        b = float(str(linear_fit(df.iloc[i])).strip().split(' ')[3])
        adict.update({df.index[i]: a})
        bdict.update({df.index[i]: b})
    aser = pd.Series(adict, name="a")
    bser = pd.Series(bdict, name="b")
    data = pd.DataFrame([aser, bser]).T
    write_excel(data, 'out_data/LinearFit_Trird.xlsx')
    
def main():
    start = time.clock()
    generate_data()
    gen_first = (time.clock() - start)
    print("Start Calc ...")
    calc_总的无季度得分()
    calc_总的有季度得分()
    calc_first = (time.clock() - gen_first)
    pre_data_generate()
    gen_sec = (time.clock() - calc_first)
    print("Start Calc ...")
    calc_第二题得分()
    calc_sec = (time.clock() - gen_sec)
    比例型得分打表()
    elapsed = (time.clock() - start)
    print("Generator First Data used:",gen_first)
    print("Calculate First Data used:",calc_first)
    print("Generator Second Data used:",gen_sec)
    print("Calculate Second Data used:",calc_sec)
    print("Full Time used:",elapsed)
    