import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def save_frames_as_video(frames_list, output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames_list:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()
    print('Saved video successfully in',output_path)

class Curbside_detection():
    def __init__(self, path):
        self.path=path
        _, ext = os.path.splitext(path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if ext.lower() in image_extensions:
            self.type = 'image'
        elif ext.lower() in video_extensions:
            self.type = 'video'

        if self.type == 'image':
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"无法读取图像路径: {path}")
            self.origin_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.gray_image = cv2.cvtColor(self.origin_image, cv2.COLOR_RGB2GRAY)
            self.gray_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        elif self.type == 'video':
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print("Error: Could not open video.")
            else:
                self.frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(self.frames,self.fps,self.width , self.height)
            frames_list = []
            while (cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame)
            cap.release()
            self.origin_video = frames_list
            self.gray_video = []
            for frame in frames_list:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                self.gray_video.append(gray_frame)

    def Edge_extraction(self,edge_type):
        if self.type == 'image':
            """Sobel方法"""
            if edge_type == "sobel":
                sobel_x = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = cv2.magnitude(sobel_x, sobel_y)
                self.Edge = np.uint8(sobel_combined)
            """Canny方法"""
            if edge_type == "canny":
                self.Edge = cv2.Canny(self.gray_image, 40, 60, apertureSize=3)
            titles = ["Original Image", "Canny Edge Image"]
            images = [self.origin_image, self.Edge]
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                if i == 0:
                    plt.imshow(images[i])
                else:
                    plt.imshow(images[i], cmap='gray')
                plt.title(titles[i])
                plt.axis('off')
            plt.show()

        elif self.type == 'video':
            self.edge_video=[]
            for gray_frame in self.gray_video:
                """Sobel方法"""
                if edge_type == "sobel":
                    sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
                    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
                    Edge = np.uint8(sobel_combined)
                """Canny方法"""
                if edge_type == "canny":
                    Edge = cv2.Canny(gray_frame, 40, 60, apertureSize=3)
                self.edge_video.append(Edge)

    def Mean_Binarization(self):
        if self.type == 'image':
            """Otsu's 二值化方法"""
            _, self.image = cv2.threshold(self.Edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plt.imshow(self.image, cmap='gray')
            plt.title('Otsu Binarization')
            plt.axis('off')
            plt.show()

        elif self.type == 'video':
            self.frame_video=[]
            for frame_edge in self.edge_video:
                _, image = cv2.threshold(frame_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.frame_video.append(image)

    def Hough_Lines(self, output_path_video):
        if self.type == 'image':
            lines = cv2.HoughLines(self.image, rho=1, theta=np.pi / 180, threshold=340)
            if lines is not None:
                dot=[]
                y=450
                for line in lines:
                    rho, theta = line[0]
                    if round(float(theta),2) > 1.40 and round(float(theta),2) < 1.65:
                        continue
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    cv2.line(self.origin_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    dot.append(((y - y2) * (x1 - x2)) / (y1 - y2) + x2)
                # print(dot)
                dot.sort()
                result = []
                for i in range(len(dot)):
                    if not result or abs(dot[i] - result[-1]) > 3:
                        result.append(dot[i])
                # print(result)
                center = (int(result[1]), y)
                cv2.circle(self.origin_image, center, 15, (255, 0, 0), -1)
                plt.imshow(self.origin_image)
                plt.title('Hough Lines')
                plt.axis('off')
                plt.show()
            else:
                print("未检测到任何直线")

        elif self.type == 'video':
            for frame_index in range(len(self.frame_video)):
                lines = cv2.HoughLines(self.frame_video[frame_index], rho=1, theta=np.pi / 180, threshold=340)
                if lines is not None:
                    for line in lines:
                        rho, theta = line[0]
                        # if round(float(theta), 2) > 1.40 and round(float(theta), 2) < 1.65:
                        #     continue
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * a)
                        cv2.line(self.origin_video[frame_index], (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print("未检测到任何直线")
            file_name = os.path.splitext(os.path.split(self.path)[1])[0]
            output_path = output_path_video+'/HoughLines_' + file_name + '.mp4'
            save_frames_as_video(self.origin_video, output_path, self.fps, self.width, self.height)

    def Hough_Lines_P(self, output_path_video):
        if self.type == 'image':
            lines = cv2.HoughLinesP(self.image, rho=1, theta=np.pi / 180, threshold=320, minLineLength=290, maxLineGap=8)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # if (y1-y2)/(x1-x2) <= 0 :
                    #     continue
                    cv2.line(self.origin_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plt.imshow(self.origin_image)
                plt.title('Probabilistic Hough Lines')
                plt.axis('off')
                plt.show()
            else:
                print("未检测到任何直线")

        elif self.type == 'video':
            center=[466]
            for frame_index in range(len(self.frame_video)):
                lines = cv2.HoughLinesP(self.frame_video[frame_index], rho=1, theta=np.pi / 180, threshold=100, minLineLength=200,maxLineGap=8)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # if (y1 - y2) / (x1 - x2) <= 0.2 or abs(x1-x2)<10:
                        #     continue
                        cv2.line(self.origin_video[frame_index], (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print("未检测到任何直线")
            file_name = os.path.splitext(os.path.split(self.path)[1])[0]
            output_path = output_path_video+'/HoughLinesP_'+file_name+'.mp4'
            save_frames_as_video(self.origin_video, output_path, self.fps, self.width, self.height)

# path = 'Data/image/frame_0114.jpg'
path = 'Data/4.mp4'
edge_core='sobel'
# hough_type='Hough_Lines'
hough_type='Hough_Lines_P'
output_path='result'

a = Curbside_detection(path)    #初始化图片、视频
a.Edge_extraction(edge_core)   #边缘提取
a.Mean_Binarization()   #二值化
if hough_type == 'Hough_Lines_P':
    a.Hough_Lines_P(output_path)
elif hough_type == 'Hough_Lines':
    a.Hough_Lines(output_path)
