#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import cv2
import math
from PIL import Image
import statistics
import collections
import numpy as np
from matplotlib import pyplot as plt

def pattern_matching(x_past, y_past, drc_past, img_path):
    time_start = time.time()
    akaze = cv2.AKAZE_create()
    #akaze = cv2.ORB_create()

    l_past = 0  # 撮影感覚で進んだ距離
    x_past = x_past + l_past*math.cos(drc_past)
    y_past = y_past + l_past*math.sin(drc_past)

    # クエリ画像を読み込んで特徴量計算
    expand_template = 4  # 重要パラメータ 1
    template_temp = cv2.imread('../pic/img_paper2.png',0)
    template_img = template_temp
    # template_img = cv2.resize(template_img, None, fx = expand_template, fy = expand_template)
    template_img = cv2.resize(template_img, (int(template_img.shape[0] / expand_template), int(template_img.shape[1] / expand_template)))
    height, width = template_img.shape[:2]
    kp_temp, des_temp = akaze.detectAndCompute(template_img, None)

    # マップ画像を読み込んで特徴量計算
    expand_sample = 1  # 重要パラメータ 2
    trim_size = 150
    sample_img = cv2.imread('../pic/field_4.png', 0)
    sample_img = cv2.resize(sample_img, None, fx = expand_sample, fy = expand_sample)
    x_past = int(x_past * expand_sample)
    y_past = int(y_past * expand_sample)
    trim_size = int(trim_size * expand_sample)
    
    top = y_past - trim_size
    if top < 0:
        top = 0
    bottom = y_past + trim_size
    if bottom > sample_img.shape[0]:
        bottom = sample_img.shape[0]
    left = x_past - trim_size
    if left < 0:
        left = 0
    right = x_past + trim_size
    if right > sample_img.shape[1]:
        right = sample_img.shape[1]

    sample_img = sample_img[top:bottom, left:right]
    # sample_img = sample_img[350:750, 550:950]
    kp_samp, des_samp = akaze.detectAndCompute(sample_img, None)
    print('feature calculation: ', time.time() - time_start)

    # 特徴量マッチング実行
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_temp, des_samp, k=2)
    print('feature matching: ', time.time() - time_start)

    # マッチング精度が高いもののみ抽出
    ratio = 0.9  # 重要パラメータ 0.6
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    good = sorted(good, key = lambda x : x[0].distance)
    #print(good)
    point_num = 10  # 重要パラメータ 7
    if len(good) < point_num:
        point_num = len(good)

    # マッチング結果の描画
    result_img = cv2.drawMatchesKnn(template_img, kp_temp, sample_img, kp_samp, good[:point_num], None, flags=0) 
    # img3 = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img3)
    # plt.show()
    print('draw mid result: ', time.time() - time_start)

    # exit()

    ### これ以降位置と向きの計算

    # q, temp = カメラ画像
    # t, samp = マップ
    q_kp = []
    t_kp = []

    # 2つの画像で対応するキーポイントを抽出
    for p in good[:point_num]:
        q_kp.append(kp_temp[p[0].queryIdx])
        t_kp.append(kp_samp[p[0].trainIdx])

    # 上位のキーポイントの相対関係を全て調べて多数決を取ることでノイズに強くする
    deg_cand = np.zeros((point_num, point_num))  # 点i, jの相対角度と相対長さを格納する配列
    len_cand = np.zeros((point_num, point_num))

    for i in range(point_num):
        for j in range(i+1, point_num):
            # クエリ画像から特徴点間の角度と距離を計算
            q_x1, q_y1 = q_kp[i].pt
            q_x2, q_y2 = q_kp[j].pt
            q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
            q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)

            # マップ画像から特徴点間の角度と距離を計算
            t_x1, t_y1 = t_kp[i].pt
            t_x2, t_y2 = t_kp[j].pt
            t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
            t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

            #print(q_x1, q_y1, q_x2, q_y2, q_deg, q_len)
            #print(t_x1, t_y1, t_x2, t_y2, t_deg, t_len)

            # 2つの画像の相対角度と距離
            deg_value = q_deg - t_deg
            if deg_value < 0:
                deg_value += 360
            size_rate = q_len/t_len

            deg_cand[i][j] = deg_value
            deg_cand[j][i] = deg_value
            len_cand[i][j] = size_rate
            len_cand[j][i] = size_rate


    #print(deg_cand)
    #print(len_cand)

    # 多数決を取るための関数
    # ある点iについて，j, kとの相対関係が一致するかを各jについて調べる
    cand_count = np.zeros((point_num, point_num))
    for i in range(len(deg_cand)):
        for j in range(len(deg_cand)):
            for k in range(len(deg_cand)):
                deg_dif = np.abs(deg_cand[i][k] - deg_cand[i][j])
                size_dif = np.abs(len_cand[i][k] - len_cand[i][j])
                if len_cand[i][k] < 0.5 or len_cand[i][j] < 0.5:  # 明らかに違う比率の結果を弾く
                    continue
                if deg_dif <= deg_cand[i][j]*0.05 and size_dif <= len_cand[i][j]*0.05:  # 重要パラメータ
                    cand_count[i][j] += 1

    #print(cand_count)
    if np.max(cand_count) == 1:  # どの2点も同じ相対関係になかった場合
        print("no matching point")
    maxidx = np.unravel_index(np.argmax(cand_count), cand_count.shape)  # もっとも多く相対関係が一致する2点を取ってくる
    deg_value = deg_cand[maxidx]
    size_rate = len_cand[maxidx]
    print(maxidx, deg_value, size_rate)

    a = maxidx[0]
    b = maxidx[1]
    print(a,b)  # この2点が，クエリ画像とマップ画像上で一致している点とみなせる（点のindex）
    print('point calculation: ', time.time() - time_start)

    q_x1, q_y1 = q_kp[a].pt
    q_x2, q_y2 = q_kp[b].pt
    q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
    q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)
    t_x1, t_y1 = t_kp[a].pt
    t_x2, t_y2 = t_kp[b].pt
    t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
    t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

    #print(q_x1, q_y1, q_x2, q_y2, q_deg, q_len)
    #print(t_x1, t_y1, t_x2, t_y2, t_deg, t_len)

    # クエリ画像の1点目とクエリ画像の中心の相対的な関係
    q_xcenter = width/2
    q_ycenter = height/2
    q_cdeg = math.atan2(q_ycenter - q_y1, q_xcenter - q_x1) * 180 / math.pi
    q_clen = math.sqrt((q_xcenter - q_x1) ** 2 + (q_ycenter - q_y1) ** 2)
    #print(q_xcenter, q_ycenter, q_cdeg, q_clen)

    # 上の関係をマップ画像上のパラメータに変換
    t_center_deg = q_cdeg - deg_value
    t_center_len = q_clen/size_rate
    #print(t_center_deg, t_center_len)

    # 中心点のマップ画像上での位置
    t_center_rad = t_center_deg * math.pi / 180
    t_xcenter = t_x1 + t_center_len * math.cos(t_center_rad)
    t_ycenter = t_y1 + t_center_len * math.sin(t_center_rad)
    #print(t_center_rad, math.cos(t_center_rad), math.sin(t_center_rad), t_xcenter, t_ycenter)

    # 中心点描画
    cv2.circle(result_img, (int(t_xcenter) + width, int(t_ycenter)), 20, color=(0, 0, 255), thickness=-1)
    #print(int(t_xcenter) + width, int(t_ycenter))

    # 向きの計算，矢印描画
    deg_front = - deg_value * math.pi / 180 - math.pi/2
    q_xfront = t_xcenter + 200 * math.cos(deg_front)
    q_yfront = t_ycenter + 200 * math.sin(deg_front)
    cv2.arrowedLine(result_img, (int(t_xcenter) + width, int(t_ycenter)), (int(q_xfront) + width, int(q_yfront)), color=(255, 0, 0), thickness=15)

    print('final prediction -> center: ({},{}), direction: {}°'.format(t_xcenter/expand_sample, t_ycenter/expand_sample, 360 - deg_value))
    print('final time: ', time.time() - time_start)

    print((t_xcenter + left)/expand_sample, (t_ycenter + top)/expand_sample, deg_value)

    x_current = int((t_xcenter + left)/expand_sample * 1800/2657)
    y_current = int((t_ycenter + top)/expand_sample * 1800/2657)
    
    if deg_value >= 180:
        drc_current = (360 - deg_value) * math.pi / 180
    else:
        drc_current = - deg_value * math.pi / 180

    print(x_current, y_current, drc_current)

    img3 = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()

    return x_current, y_current, drc_current

if __name__ == "__main__":
    x_past = 400
    y_past = 300
    drc_past = math.pi
    x_current, y_current, drc_current = pattern_matching(x_past, y_past, drc_past, None)
