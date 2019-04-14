import cv2
import numpy as np
import os
from create_base import *
import time

def get_contours(filename,pic_path):
    #pic_path = "./tmp/tmp.jpg"
    y,sr = get_cqt_pic(filename,pic_path)
    rms = librosa.feature.rmse(y=y)[0]
    length = len(rms)
    img = cv2.imread(pic_path)
    # kernel = np.ones((2, 2), np.uint8)*np.max(img)  # 全为1的过滤数组
    # # 膨胀(去黑小点)
    # img = cv2.dilate(img, kernel)  # 膨胀

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


    _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
    # draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
    # draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
    #draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 1)
    draw_img3 = img.copy()
    contours_dict = {}
    right_lines = []
    right_points = []
    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        right_points.append(x + w)
    for i in range(0,len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        #cv2.rectangle(draw_img3, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if w * h > 30:
            cv2.rectangle(draw_img3, (x,y), (x+w,y+h), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw_img3, str(x) + ',' + str(y), (x,y), font, 0.4, (255, 255, 255), 1)
            #print("START x,w is {},{}".format(x, x + w))
            #根据左边线判断
            is_exist = False
            tmp = []
            for j in range(x-5,x+5):
                if j in contours_dict.keys():
                    #print("1. x,w is {},{}".format(x,x+w))
                    tmp.append(contours_dict.get(j))
                    tmp.append([x, y, w, h])
                    contours_dict.update({int((x + j)/2):tmp})
                    #添加右边线
                    if w > 10 and h > 10 and check_right_line(right_points,x + w):
                        right_lines.append(x + w)
                    is_exist = True
                    break
            if is_exist is False:
                tmp.append([x, y, w, h])
                contours_dict.update({x:tmp})
                #print("2. x,w is {},{}".format(x, x + w))
            # 根据右边线判断
            # is_exist = False
            # tmp = []
            # for j in range(x + w - 3, x + w + 3):
            #     if j in contours_dict.keys():
            #         tmp.append(contours_dict.get(j))
            #         tmp.append(contours[i])
            #         contours_dict.update({int((x + w + j) / 2): tmp})
            #         is_exist = True
            #         break
    find_notes = find_note_by_contours(contours_dict)
    for x in find_notes:
        cv2.line(draw_img3, (x, 10), (x, 200), (0, 255, 0), 2)  # 绿色，3个像素宽度
    right_lines.sort()
    if len(right_lines)>0:
        right_lines = del_overrate(right_lines)
        right_lines = check_with_start(find_notes, right_lines)
    for x in right_lines:
        cv2.line(draw_img3, (x, 10), (x, 200), (255, 255, 0), 2)  # 绿色，3个像素宽度
        find_notes.append(x)
    find_notes.sort()
    return draw_img3,img,find_notes,length

def get_note_lines(img,find_notes,lenght):
    width = img.shape[1]
    result = []
    for x in find_notes:
        tmp = int(x * lenght /width)
        result.append(tmp)
    return result

def find_note_by_contours(contours_dict):
    result = []
    for key in contours_dict.keys():
        tmp = contours_dict.get(key)
        if len(tmp)>=2:
            result.append(key)
        elif tmp[0][2]*tmp[0][3] > 50 and tmp[0][2]*tmp[0][3] < 10000:
            result.append(key)
    result.sort()
    result = del_overrate(result)
    return result

def del_overrate(result):
    new_result = [result[0]]
    last = result[0]
    for i in range(1, len(result)):
        if result[i] - last > 3:
            new_result.append(result[i])
            last = result[i]
    return new_result

def check_with_start(start_nots,right_lines):
    result = []
    for r in right_lines:
        tmp = [np.abs(r - x) for x in start_nots]
        if np.min(tmp) > 5:
            result.append(r)
    return result


def check_right_line(right_points,right):
    offset = [x - right for x in right_points]
    find_nearly = [x for x in offset if x<=1]
    if len(find_nearly)>3:
        return True
    else:
        return False

def get_cqt_pic(filename,pic_path):
    _,y,sr = draw_on_cqt(filename,pic_path, False)
    plt.close()
    return y,sr

if __name__ == "__main__":
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音4(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律八（2）（60）.wav'
    start_time = time.time()
    #get_cqt_pic(filename)
    end_time = time.time()
    print("runing is {}".format(end_time - start_time))
    # filename = 'E:/test_image/n/A/50.jpg'
    # filename = 'e:/test_image/n/D/4.jpg'
    # filename = 'e:/test_image/n/C/54.jpg'
    pic_path = "./tmp/tmp.jpg"
    draw_img3,img,note_lines,length = get_contours(filename,pic_path)
    #cv2.imshow("img", img)
    # cv2.imshow("draw_img0", draw_img0)
    # cv2.imshow("draw_img1", draw_img1)
    # cv2.imshow("draw_img2", draw_img2)
    cv2.imshow("draw_img3", draw_img3)
    print("width,height is {},{}".format(img.shape[1],img.shape[0]))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite( "f:/test1111..jpg",draw_img3)


    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    dir_list = ['e:/test_image/n/B/']
    dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/c/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            draw_img3, img = get_contours(dir + filename)

            grade = 'B'
            cv2.imwrite(result_path + grade + "/" + filename, draw_img3)