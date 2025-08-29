#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#coding=utf-8

import re
import build
from Bio import SeqIO
from sys import argv
import datetime
import sys
sys.setrecursionlimit(10000) 
seqs = {}
for seq_record in SeqIO.parse(argv[2], "fasta"):
    seqs[seq_record.id] = seq_record.seq._data

def secondbubble(bubble, seq1, seq2):
    bubble2 = {}
    bubble2.update(bubble)
    for x in bubble:
        if bubble[x]["length2"] > 5 and bubble[x]["length1"] > 5:
            bbbb = build.secondconstruteDG(
                bubble[x]["start_pos1"],
                bubble[x]["start_pos2"],
                x,
                seq1[int(bubble[x]["start_pos1"]) - 1:int(bubble[x]["end_pos1"])],
                seq2[int(bubble[x]["start_pos2"]) - 1:int(bubble[x]["end_pos2"])]
            )
            bubble2.update(bbbb)
    return bubble2

af = open(argv[1] + ".AF", "w")
al = open(argv[1] + ".AL", "w")
mx = open(argv[1] + ".MX", "w")
snp = open(argv[1] + ".snp", "w")
bb = open(argv[1] + ".bubble", "w")
ioe = open(argv[1], "r")
done = open(argv[1] + ".done", "w")

# 写入 .bubble 文件表头
bb.write("QueryName,SubjectName,BubbleID,Flag,SNP_num,NewUp,NewAA,NewDown,Start_pos1,End_pos1,Start_pos2,End_pos2,Length1,Length2\n")

while True:
    lines = ioe.readlines(100000)
    if not lines:
        break
    for line in lines:
        line = line.strip('\n')
        hang = line.split(',')
        queryName = hang[0]
        subjectName = hang[1]
        bothname = queryName + "," + subjectName
        print(bothname, file=done)
        if queryName != subjectName:
            bubble = build.construteDG(seqs[queryName], seqs[subjectName])
            bubble = secondbubble(bubble, seqs[queryName], seqs[subjectName])
            flag = "0"
            snp_num = 0
            outline = ''
            outline_af = ''
            outline_al = ''
            outline_mx = ''
            bridge = 0
            lastendpos = 0
            if len(bubble) == 1 and bubble["bubble1"]["length2"] > 10 and bubble["bubble1"]["length1"] > 10:
                if bubble["bubble1"]["length1"] / len(seqs[queryName]) <= 0.3 or bubble["bubble1"]["length2"] / len(seqs[subjectName]) <= 0.3:
                    if "Q" in bubble["bubble1"]["start_node"]:
                        flag = "af"
                        outline_af = bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "\n"
                    elif "Q" in bubble["bubble1"]["end_node"]:
                        flag = "al"
                        outline_al = bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "\n"
                    else:
                        flag = "mx"
                        outline_mx = bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "\n"
                    bridge = len(seqs[queryName]) - bubble["bubble1"]["length1"]
            else:
                for x in bubble:
                    bridge = bridge + int(bubble[x]["start_pos1"]) - lastendpos
                    lastendpos = int(bubble[x]["end_pos1"])
                    newup = ""
                    newaa = ""
                    newdown = ""
                    if bubble[x]["length1"] == 0 and bubble[x]["length2"] > 2 and "Q" not in bubble[x]["start_node"] and "Q" not in bubble[x]["end_node"]:
                        flag = "bubble" + "_" + str(bubble[x]["length1"]) + "_" + str(bubble[x]["length2"])
                        newup = seqs[subjectName][max(0, int(bubble[x]["start_pos2"]) - 50):int(bubble[x]["start_pos2"])]
                        newaa = seqs[subjectName][int(bubble[x]["start_pos2"]):int(bubble[x]["end_pos2"])]
                        newdown = seqs[subjectName][int(bubble[x]["end_pos2"]):int(bubble[x]["end_pos2"]) + 50]
                        outline = bothname + "," + newup + "," + newaa + "," + newdown + ',' + str(bubble[x]["start_pos1"]) + ',' + str(bubble[x]["end_pos1"]) + ',' + str(bubble[x]["start_pos2"]) + ',' + str(bubble[x]["end_pos2"]) + ',' + str(len(seqs[queryName])) + ',' + str(len(seqs[subjectName])) + "\n"
                    elif bubble[x]["length2"] == 0 and bubble[x]["length1"] > 2 and "Q" not in bubble[x]["start_node"] and "Q" not in bubble[x]["end_node"]:
                        flag = "bubble" + "_" + str(bubble[x]["length1"]) + "_" + str(bubble[x]["length2"])
                        newup = seqs[queryName][max(0, int(bubble[x]["start_pos1"]) - 50):int(bubble[x]["start_pos1"])]
                        newaa = seqs[queryName][int(bubble[x]["start_pos1"]):int(bubble[x]["end_pos1"])]
                        newdown = seqs[queryName][int(bubble[x]["end_pos1"]):int(bubble[x]["end_pos1"]) + 50]
                        outline = bothname + "," + newup + "," + newaa + "," + newdown + ',' + str(bubble[x]["start_pos1"]) + ',' + str(bubble[x]["end_pos1"]) + ',' + str(bubble[x]["start_pos2"]) + ',' + str(bubble[x]["end_pos2"]) + ',' + str(len(seqs[queryName])) + ',' + str(len(seqs[subjectName])) + "\n"
                    elif bubble[x]["length2"] == 1 and bubble[x]["length1"] == 1:
                        snp_num = snp_num + 1
                    elif bubble[x]["length2"] > 10 and bubble[x]["length1"] > 10:
                        if "Q" in bubble["bubble1"]["start_node"]:
                            flag = "af"
                            outline_af = outline_af + bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "," + "\n"
                        elif "Q" in bubble["bubble1"]["end_node"]:
                            flag = "al"
                            outline_al = outline_al + bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "," + "\n"
                        else:
                            flag = "mx"
                            outline_mx = outline_mx + bothname + "," + str(bubble["bubble1"]["start_pos1"]) + "," + str(bubble["bubble1"]["end_pos1"]) + "," + str(bubble["bubble1"]["start_pos2"]) + "," + str(bubble["bubble1"]["end_pos2"]) + "," + "\n"
                    # 输出到 .bubble 文件，包含剪切序列和上下游 50bp
                    print(f"{queryName},{subjectName},{x},{flag},{snp_num},{newup},{newaa},{newdown},{bubble[x]['start_pos1']},{bubble[x]['end_pos1']},{bubble[x]['start_pos2']},{bubble[x]['end_pos2']},{bubble[x]['length1']},{bubble[x]['length2']}", file=bb)
            bridge = bridge + len(seqs[queryName]) - lastendpos
            coverage = bridge / min(len(seqs[queryName]), len(seqs[subjectName]))
            if flag != "0" and snp_num > 2 and coverage > 0.6:
                print(snp_num, outline, end='', file=snp)
            if flag != "0" and snp_num < 3 and coverage > 0.6:
                if ",," not in outline:
                    print(outline, end='')
                print(outline_af, end='', file=af)
                print(outline_al, end='', file=al)
                print(outline_mx, end='', file=mx)

af.close()
al.close()
mx.close()
snp.close()
bb.close()
ioe.close()
done.close()