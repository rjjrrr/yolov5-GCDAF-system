# -*- coding: utf-8 -*-
import shutil
import sqlite3
import time

from PIL import ImageFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QHeaderView, QAbstractItemView, QFileDialog, QTableWidgetItem, \
    QMessageBox
import sys

from UI.precess_bar import ProgressBar
''' 
本系统基于YOLOv5开源检测框架结构实现，模型推理流程参考Ultralytics YOLOv5项目，原始代码链接：https://github.com/ultralytics/yolov5；
本项目基于公开框架与主流技术路线的自主开发与扩展，检测逻辑参考了yolo检测思想；
本项目部分参考了网络上成熟的检测系统开发思路；
在此基础上，本作者引入并设计GCDAF结构进行算法改进，并开发PyQt界面与系统功能扩展；
'''
sys.path.append('UI')
from UI.UiMain import Ui_MainWindow
from UI.login import Ui_MainWindow as loginWindow
from UI.register import Ui_MainWindow as registerWindow
from PyQt5.QtCore import Qt, QCoreApplication, QTimer, QThread, pyqtSignal
import warnings
warnings.filterwarnings("ignore")
import detect_tools as tools
import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
from uuid import uuid4
import math
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (Profile, check_file, check_img_size, check_imshow, cv2,increment_path, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from PyQt5.QtWidgets import QTextBrowser
from PyQt5 import QtCore
import os
import platform
import subprocess
from PyQt5.QtCore import pyqtSignal
import datetime
def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])
class MainWindow(QMainWindow):
    log_signal = pyqtSignal(str)
    update_nums_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()
        self.save_path = 'save_data'
        os.makedirs(self.save_path, exist_ok=True)  # ✅ 确保目录存在
        self.model_path = 'runs/train/exp8/weights/best.pt'
        self.model = None
        if Path(self.model_path).is_file():
            self.init_model(self.model_path)
        self.purge_run_folder("./runs/detect")
        # 分类统计图嵌入区域（表格下方，右下角）
        self.figure = Figure(figsize=(4, 1.5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setGeometry(780, 780, 420, 120)
        self.canvas.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.canvas.show()

        self.log_browser = QTextBrowser(self)
        self.log_browser.setGeometry(50, 800, 580, 80)
        self.log_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ccc;
                border-radius: 6px;
                background-color: #ffffff;
                padding: 4px;
                font-size: 12px;
                font-family: "Microsoft YaHei";
            }
        """)
        self.log_browser.setPlaceholderText("📋 检测历史日志将在此显示…")
        self.log_browser.setReadOnly(True)
        self.log_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log_browser.raise_()  # ✅ 置顶
        self.log_browser.show()
        self.log_signal.connect(self.log_browser.append)

    def show_video_score_popup(self, message):
        QMessageBox.information(self, "视频检测评分", message)

    def write_log(self, source_name, class_count: int, target_count: int):
        if not source_name:
            source_name = "未知来源"
        now = datetime.datetime.now().strftime("%H:%M:%S")

        msg = f"[{now}] 检测 {source_name}，共检测到 {class_count} 类 {target_count} 个目标"

        if hasattr(self, "last_detect_score"):
            msg += f"，评分 {self.last_detect_score:.2f} / 100"

        print(f"[写入日志] {msg}")
        self.log_signal.emit(msg)  # ✅ 替代 .append()，避免跨线程操作

    def initMain(self):
        self.ui.SaveBtn.setEnabled(False)  # 默认禁用
        # === UI参数 ===
        self.show_width: int = 560
        self.show_height: int = 360
        self.org_path: str = None
        # === 状态变量 ===
        self.is_camera_open: bool = False
        self.cap = None
        self.stream_handler = None  # 帧处理
        # === 表格设置 ===
        table = self.ui.tableWidget
        table.verticalHeader().setDefaultSectionSize(40)
        table.horizontalHeader().setSectionResizeMode(1)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        # === UI状态初始化 ===
        self.ui.SaveBtn.setEnabled(False)
        self.ui.label_show.clear()
        self.ui.label_show_2.clear()
        #self.ui.label_nums.setText("0")
        self.ui.time_lb.setText("0.000 s")
        # === 欢迎提示 ===
        QMessageBox.information(self, "欢迎使用", "检测系统已准备就绪，开始选择图像或视频进行分析吧！")
        # === 表格列宽恢复 ===
        config_path = "config/table_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    import json
                    widths = json.load(f)
                    for i, w in enumerate(widths):
                        self.ui.tableWidget.setColumnWidth(i, w)
            except Exception as e:
                print(f"[WARNING] 加载表格列宽失败: {e}")

    def init_model(self, weights_path='runs/train/exp8/weights/best.pt'):
        """加载检测模型并完成预热操作"""
        # 推理分辨率设定
        input_size = (640, 640)
        validated_size = check_img_size(input_size, s=1)

        # 自动检测可用设备（GPU优先）
        computation_device = select_device('')

        # 模型加载配置
        config_file = ROOT / 'data/coco128.yaml'
        self.model = DetectMultiBackend(weights_path, device=computation_device, data=config_file)

        # 创建空输入张量进行推理预热
        warmup_input = torch.zeros(1, 3, *validated_size, device=computation_device)
        self.model(warmup_input)

    def parse_opt(self, data_path):
        parser = argparse.ArgumentParser(description="Object Detection Configuration")

        # 参数配置表（更易维护、避免重复代码）
        param_config = {
            "--weights": {"nargs": '+', "type": str, "default": str(ROOT / 'runs/train/exp8/weights/best.pt'),
                          "help": "Path to model weights"},
            "--source": {"type": str, "default": data_path, "help": "Source: file/dir/URL"},
            "--data": {"type": str, "default": str(ROOT / 'data/coco128.yaml'), "help": "Data YAML path"},
            "--imgsz": {"nargs": '+', "type": int, "default": [640], "help": "Input size (h,w)"},
            "--conf-thres": {"type": float, "default": float(self.ui.conf_lineEdit.text()),
                             "help": "Confidence threshold"},
            "--iou-thres": {"type": float, "default": float(self.ui.iou_lineEdit.text()), "help": "IoU threshold"},
            "--max-det": {"type": int, "default": 1000, "help": "Max detections per image"},
            "--device": {"default": "", "help": "Device ID or 'cpu'"},
            "--view-img": {"action": 'store_true', "help": "Display results"},
            "--save-txt": {"action": 'store_true', "help": "Save results to txt"},
            "--save-conf": {"action": 'store_true', "help": "Save confidences to txt"},
            "--save-crop": {"action": 'store_true', "help": "Save cropped boxes"},
            "--nosave": {"action": 'store_true', "help": "Don't save output"},
            "--classes": {"nargs": '+', "type": int, "help": "Filter by class"},
            "--agnostic-nms": {"action": 'store_true', "help": "Class-agnostic NMS"},
            "--augment": {"action": 'store_true', "help": "Use augmented inference"},
            "--visualize": {"action": 'store_true', "help": "Visualize features"},
            "--update": {"action": 'store_true', "help": "Update all models"},
            "--project": {"default": str(ROOT / 'runs/detect'), "help": "Save project root"},
            "--name": {"default": 'exp', "help": "Experiment name"},
            "--exist-ok": {"action": 'store_true', "help": "Allow existing project"},
            "--line-thickness": {"type": int, "default": 3, "help": "Box line thickness"},
            "--hide-labels": {"action": 'store_true', "help": "Hide class labels"},
            "--hide-conf": {"action": 'store_true', "help": "Hide confidence scores"},
            "--half": {"action": 'store_true', "help": "Use FP16 inference"},
            "--dnn": {"action": 'store_true', "help": "Use OpenCV DNN"},
            "--vid-stride": {"type": int, "default": 1, "help": "Video frame stride"},
        }

        for arg, options in param_config.items():
            parser.add_argument(arg, **options)

        opt = parser.parse_args()
        if len(opt.imgsz) == 1:
            opt.imgsz *= 2
        return opt

    def run(self, **kwargs):


        dataset, save_dir, bs = self._prepare_dataset_and_dirs(kwargs)
        self._ensure_conf_history()

        boxes_all, labels_all, conf_vals, conf_strs, images_all = [], [], [], [], []
        final_img = None

        for path, input_arr, raw_img, *_ in dataset:
            input_tensor = self._preprocess_input(input_arr)
            preds = self.model(input_tensor, augment=kwargs['augment'], visualize=kwargs['visualize'])
            results = non_max_suppression(preds, kwargs['conf_thres'], kwargs['iou_thres'],
                                          kwargs['classes'], kwargs['agnostic_nms'], kwargs['max_det'])

            for det in results:
                annotated_img = raw_img.copy()
                annotator = Annotator(annotated_img, line_width=kwargs['line_thickness'], pil=True)

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(input_tensor.shape[2:], det[:, :4], raw_img.shape).round()

                    for *xyxy, conf, cls_id in reversed(det):
                        label = self.model.names[int(cls_id)]
                        label_text = self._format_label(xyxy, conf, int(cls_id), label, kwargs['hide_labels'],
                                                        kwargs['hide_conf'])
                        annotator.box_label(xyxy, label_text, color=colors(int(cls_id), True))

                        self._record_detection(xyxy, conf, label, annotated_img,
                                               boxes_all, labels_all, conf_vals, images_all)

                        if kwargs['save_txt']:
                            self._save_txt_label(path, raw_img, xyxy, conf, int(cls_id), save_dir, kwargs['save_conf'])

                        if kwargs['save_crop']:
                            save_one_box(xyxy, annotated_img.copy(),
                                         file=save_dir / 'crops' / label / f"{Path(path).stem}.jpg", BGR=True)

                final_img = annotator.result()
                if not kwargs['nosave']:
                    cv2.imwrite(str(save_dir / Path(path).name), final_img)

        conf_strs = [f"{v * 100:.2f} %" for v in conf_vals]
        return boxes_all, labels_all, conf_strs, conf_vals, images_all, final_img

    def _prepare_dataset_and_dirs(self, kwargs):
        source = str(kwargs['source'])
        is_stream = source.isnumeric() or source.startswith(('rtsp://', 'http', 'screen'))

        dataset = LoadStreams(source, img_size=kwargs['imgsz'], stride=self.model.stride,
                              auto=self.model.pt, vid_stride=kwargs['vid_stride']) if is_stream else \
            LoadImages(source, img_size=kwargs['imgsz'], stride=self.model.stride,
                       auto=self.model.pt, vid_stride=kwargs['vid_stride'])
        bs = len(dataset) if is_stream else 1

        save_dir = increment_path(Path(kwargs['project']) / kwargs['name'], exist_ok=kwargs['exist_ok'])
        (save_dir / 'labels' if kwargs['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)
        self.model.warmup(imgsz=(1 if self.model.pt else bs, 3, *kwargs['imgsz']))

        return dataset, save_dir, bs

    def _ensure_conf_history(self):
        if not hasattr(self, 'conf_history'):
            self.conf_history = {}

    def _preprocess_input(self, input_arr):
        tensor = torch.from_numpy(input_arr).to(self.model.device).float() / 255.0
        if self.model.fp16:
            tensor = tensor.half()
        return tensor.unsqueeze(0) if tensor.ndim == 3 else tensor

    def _get_stability_flag(self, cls, xyxy, conf):
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        key = (cls, cx // 10 * 10, cy // 10 * 10)

        self.conf_history.setdefault(key, []).append(conf.item())
        self.conf_history[key] = self.conf_history[key][-8:]

        hist = self.conf_history[key]
        mean = sum(hist) / len(hist)
        std = (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5

        return 'stable' if std < 0.07 else 'unstable'

    def _format_label(self, xyxy, conf, cls, label, hide_labels, hide_conf):
        if hide_labels:
            return None
        volatility = self._get_stability_flag(cls, xyxy, conf)
        return label if hide_conf else f"{label} {conf:.2f} {volatility}"

    def _record_detection(self, xyxy, conf, label, image, boxes, labels, confs, images):
        boxes.append([float(x.item()) for x in xyxy])
        labels.append(label)
        confs.append(float(conf.item()))
        images.append(image)

    def _save_txt_label(self, path, img, xyxy, conf, cls, save_dir, save_conf):
        txt_file = save_dir / 'labels' / f"{Path(path).stem}.txt"
        txt_file.parent.mkdir(parents=True, exist_ok=True)
        norm_box = xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(img.shape)[[1, 0, 1, 0]]
        line = (cls, *norm_box.view(-1).tolist(), conf.item()) if save_conf else (cls, *norm_box.view(-1).tolist())
        with open(txt_file, 'a') as f:
            f.write(' '.join(map(str, line)) + '\n')

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.picuture)
        self.ui.PicBtn.setShortcut("Ctrl+O")  # 打开图片：Ctrl+O

        self.ui.VideoBtn.clicked.connect(self.vedio)
        self.ui.VideoBtn.setShortcut("Ctrl+V")  # 打开视频：Ctrl+V

        self.ui.CapBtn.clicked.connect(self.camera)
        self.ui.CapBtn.setShortcut("Ctrl+C")  # 打开摄像头：Ctrl+C

        self.ui.SaveBtn.clicked.connect(self.save)
        self.ui.SaveBtn.setShortcut("Ctrl+S")  # 保存结果：Ctrl+S

        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.ExitBtn.setShortcut("Ctrl+Q")  # 退出：Ctrl+Q

        self.ui.ModelBtn.clicked.connect(self.model)
        self.ui.ModelBtn.setShortcut("Ctrl+M")  # 加载模型：Ctrl+M

        self.ui.conf_lineEdit.editingFinished.connect(self.validate_threshold_inputs)
        self.ui.iou_lineEdit.editingFinished.connect(self.validate_threshold_inputs)

    def process_source(self, source_type, source_path=None):
        """
        source_type: 'image' | 'video' | 'camera'
        source_path: 图像或视频文件路径
        """
        if self.cap:
            self.video_stop()
        self.is_camera_open = False

        self.save_path = os.path.join(os.getcwd(), "save_data")
        os.makedirs(self.save_path, exist_ok=True)

        if source_type == 'image':
            if not source_path:
                file_path, _ = QFileDialog.getOpenFileName(self, '打开图片', './', "Image files (*.jpg *.jpeg *.png)")
                if not file_path: return
                source_path = file_path

            self.org_path = source_path
            self.org_img = tools.img_cvread(self.org_path)
            w, h = self.compute_display_size(self.org_img)
            pix_img = tools.cvimg_to_qpiximg(cv2.resize(self.org_img, (w, h)))
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            t1 = time.time()
            self.location_list, self.cls_list, self.conf_list, self.conf_vals, self.img_list, now_img = self.detect_model(
                self.org_path, show_score=False)
            t2 = time.time()
            self.ui.time_lb.setText(f'{t2 - t1:.3f} s')

            # ✅ 检测完后再单独调用评分（不影响计时）
            self.evaluate_score(self.cls_list, self.conf_vals)

            self.draw_img = now_img

            w, h = self.compute_display_size(now_img)
            pix_img = tools.cvimg_to_qpiximg(cv2.resize(now_img, (w, h)))
            self.ui.label_show_2.setPixmap(pix_img)
            self.ui.label_show_2.setAlignment(Qt.AlignCenter)

            #self.ui.label_nums.setText(str(len(self.cls_list)))
            self.ui.SaveBtn.setEnabled(True)
            self.ui.tableWidget.setRowCount(0)
            self.ui.tableWidget.clearContents()
            self.populate_result_table(self.location_list, self.cls_list, self.conf_list)
            self.plot_detection_stats(self.cls_list)
            self.write_log(os.path.basename(self.org_path), len(set(self.cls_list)), len(self.cls_list))

        elif source_type in ['video', 'camera']:
            if not source_path and source_type == 'video':
                source_path = self.get_video_path()
                if not source_path: return

            self.org_path = source_path if source_type == 'video' else None
            self.cap = cv2.VideoCapture(source_path if source_type == 'video' else 0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误",
                                    f"无法打开{'视频文件' if source_type == 'video' else '摄像头'}，请检查路径或设备连接。")
                return

            self.is_camera_open = source_type == 'camera'

            self.stream_handler = FrameStreamHandler(
                parent_ui=self,
                cap=self.cap,
                detector=self.detect_model,
                update_frame_callback=self.display_original_frame,
                time_label=self.ui.time_lb,
                result_display=self.display_detection_results
            )

            self.stream_handler.start()
            self.ui.SaveBtn.setEnabled(True)

    def display_original_frame(self, frame):
        w, h = self.compute_display_size(frame)
        pix = tools.cvimg_to_qpiximg(cv2.resize(frame, (w, h)))
        self.ui.label_show.setPixmap(pix)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def display_detection_results(self, result_img, clses, confs, boxes):
        w, h = self.compute_display_size(result_img)
        pix = tools.cvimg_to_qpiximg(cv2.resize(result_img, (w, h)))
        self.ui.label_show_2.setPixmap(pix)
        self.ui.label_show_2.setAlignment(Qt.AlignCenter)
        self.populate_result_table(boxes, clses, confs)
        self.plot_detection_stats(clses)

    def picuture(self):
        if not self._ensure_model_loaded():
            return
        self.process_source('image')

    def vedio(self):
        if not self._ensure_model_loaded():
            return
        self.process_source('video')

    def camera(self):
        if not self._ensure_model_loaded():
            return
        self.process_source('camera')

    def _ensure_model_loaded(self):
        """Require users of the public source release to select local weights."""
        if self.model is not None:
            return True
        QMessageBox.warning(self, "未加载模型", "请先点击“模型选择”并加载本地 .pt 权重文件。")
        return False

    def save(self):
        if not self.org_path or (self.cap is None and self.draw_img is None):
            QMessageBox.warning(self, "提示", "请先打开图片或视频再进行保存！")
            return

        if self.is_camera_open:
            QMessageBox.warning(self, "提示", "摄像头实时画面暂不支持保存！")
            return

        if self.cap:
            self._save_video_result()
        else:
            self._save_image_result()

    def _save_image_result(self):
        if not self.draw_img.any():
            QMessageBox.warning(self, "保存失败", "检测图像为空，无法保存")
            return

        name, ext = os.path.splitext(os.path.basename(self.org_path))
        save_name = f"{name}_detect_result{ext}"
        save_img_path = os.path.join(self.save_path, save_name)

        try:
            cv2.imwrite(save_img_path, self.draw_img)
            QMessageBox.information(self, "保存成功", f"检测结果已保存：\n{save_img_path}")
            open_folder(self.save_path)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"发生错误：\n{str(e)}")

    def _save_video_result(self):
        confirm = QMessageBox.question(
            self, "确认保存",
            "保存视频检测结果可能需要较长时间，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if confirm != QMessageBox.Yes:
            return

        self.worker = VideoDetectWorker(
            self.save_path,
            self.org_path,
            self.frame_only_detect,  # ✅ 替换为线程安全版本
            self
        )
        self.worker.score_ready.connect(self.show_video_score_popup)

        self.worker.progress_updated.connect(self.track_video_progress)
        self.worker.start()

    def model(self):
        path = self._prompt_model_path()
        if not path:
            self._notify_user("取消操作", "未选择任何模型文件。")
            return

        success = self._attempt_model_load(path)
        if success:
            self.model_path = path

    def _prompt_model_path(self):
        """弹出文件选择器，返回用户选择的模型路径"""
        return QFileDialog.getOpenFileName(
            self, "选择模型文件", "./weights", "模型文件 (*.pt)"
        )[0]

    def _attempt_model_load(self, path):
        """尝试加载模型，捕获异常并反馈用户"""
        try:
            self.init_model(path)
            self._notify_user("模型加载成功", f"已成功载入模型：{os.path.basename(path)}")
            return True
        except Exception as error:
            self._notify_user("模型加载失败", f"加载过程中出现错误：\n{str(error)}", error=True)
            return False

    def _notify_user(self, title, message, error=False):
        """根据类型弹出提示或错误信息"""
        if error:
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.information(self, title, message)

    def plot_detection_stats(self, cls_list):
        from collections import Counter
        import matplotlib
        import matplotlib.ticker as ticker
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        count = Counter(cls_list)
        labels = list(count.keys())
        values = list(count.values())

        if not labels:
            ax.text(0.5, 0.5, "无检测结果", ha='center', va='center', fontsize=12)
        else:
            # ✅ 控制高度：类别越少，越细，让条之间空隙大些
            bar_height = 0.4 if len(labels) <= 3 else 0.6

            bars = ax.barh(labels, values, color='#3a92fe', height=bar_height)
            ax.set_title("检测类别统计（单位：个）", fontsize=10, pad=10)
            ax.set_xlabel("数量", fontsize=9)
            ax.tick_params(axis='y', labelsize=9)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                        ha='left', va='center', fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()

    def video_stop(self):
        if hasattr(self, 'stream_handler') and self.stream_handler:
            self.stream_handler.stop()
            self.stream_handler = None

        if self.cap:
            self.cap.release()
            self.cap = None

        self.ui.label_show.clear()
        self.ui.label_show_2.clear()
        #self.ui.label_nums.setText("0")

    def _log(self, message):
        """记录关键步骤到控制台，并在日志区输出（若存在）"""
        print(f"[video_stop] {message}")
        if hasattr(self, 'log_browser'):
            self.log_browser.append(f"📍 {message}")

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,  # ✅ 使用主窗口作为父级，风格统一
            "选择视频文件",
            "./",
            "视频文件 (*.avi *.mp4 *.mov *.mkv)"
        )
        return file_path if file_path else None



    def _reset_table(self):
        """清空检测结果表格"""
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()



    def track_video_progress(self, current_frame, total_frames):
        """追踪视频处理进度，在首帧初始化，中途刷新 UI，末帧自动收尾，异常自动终止"""

        # 📌 首帧：初始化进度窗口
        if current_frame == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
            print(f"[ProgressFlow] 📈 检测任务已启动，准备接收帧序列。")
            if hasattr(self, 'log_browser'):
                self.log_browser.append("📈 检测任务已启动，准备接收帧序列。")

        # 📌 完成：关闭窗口 + 通知 + 打开文件夹
        if current_frame >= total_frames:
            if self.progress_bar:
                self.progress_bar.close()
            QMessageBox.information(self, "任务完成", f"视频检测结果已保存至：\n{self.save_path}")
            open_folder(self.save_path)
            print(f"[ProgressFlow] ✅ 视频检测任务圆满完成，结果已保存。")
            if hasattr(self, 'log_browser'):
                self.log_browser.append("✅ 视频检测任务圆满完成，结果已保存。")
            return

        # 📌 中断：窗口被关，线程终止
        if not (hasattr(self, 'progress_bar') and self.progress_bar.isVisible()):
            if hasattr(self, 'worker'):
                self.worker.stop()
                print(f"[VideoFlow] ⚠️ 进度窗口已关闭，检测线程已主动终止。")
                if hasattr(self, 'log_browser'):
                    self.log_browser.append("⚠️ 检测线程终止：用户关闭进度窗口。")
            return

        # 📌 更新：刷新数值显示
        ratio = min(max(current_frame / total_frames, 0), 1.0)
        percent = int(ratio * 100)
        self.progress_bar.setValue(current_frame, total_frames, percent)
        QApplication.processEvents()
        print(f"[ProgressFlow] ⏳ 当前进度：{current_frame}/{total_frames} 帧（{percent}%）")
        if hasattr(self, 'log_browser'):
            self.log_browser.append(f"⏳ 当前进度：{current_frame}/{total_frames} 帧（{percent}%）")

    def validate_threshold_inputs(self):
        """解析并验证界面上的置信度与 IoU 阈值输入，若无效则自动纠正并提示"""
        raw_conf = self.ui.conf_lineEdit.text()
        raw_iou = self.ui.iou_lineEdit.text()

        try:
            conf = float(raw_conf)
            iou = float(raw_iou)
            if not (0 <= conf <= 1 and 0 <= iou <= 1):
                raise ValueError("阈值范围超出 0~1")

        except Exception as err:
            QMessageBox.warning(self, "阈值格式错误", "请输入 0 到 1 之间的小数，例如 0.35")
            self.ui.conf_lineEdit.setText("0.35")
            self.ui.iou_lineEdit.setText("0.45")
            self.conf_thres = 0.35
            self.iou_thres = 0.45
            print(f"[Threshold] ❌ 输入非法，已重置默认值。原因：{err}")
            return

        self.conf_thres = conf
        self.iou_thres = iou
        print(f"[Threshold] ✅ 阈值已更新：Conf = {conf:.2f}, IoU = {iou:.2f}")

    def detect_model(self, source, show_score=True):
        """
        使用当前加载的检测模型对输入执行推理，可接收文件路径或 OpenCV 图像数组。
        若输入为图像数据，将写入临时文件中执行推理后再清理。
        """
        is_array_input = isinstance(source, np.ndarray)
        cleanup_required = False

        # 如果是图像数组，保存为临时图像以适配推理函数
        if is_array_input:
            temp_dir = os.path.join(os.getcwd(), "temp_input")
            os.makedirs(temp_dir, exist_ok=True)
            temp_filename = f"frame_{uuid4().hex[:6]}.jpg"
            temp_path = os.path.join(temp_dir, temp_filename)
            cv2.imwrite(temp_path, source)
            source = temp_path
            cleanup_required = True

        # 构造推理参数，执行检测
        opt = self.parse_opt(source)
        location_list, cls_list, conf_str_list, conf_raw_list, img_list, result_img = self.run(**vars(opt))
        conf_list = conf_str_list  # 仍用于 UI 表格和日志
        self.last_detect_confidences = conf_raw_list  # 视频/摄像头评分使用
        self.last_detect_classes = cls_list

        # 清除中转文件
        if cleanup_required and os.path.exists(source):
            try:
                os.remove(source)
            except Exception as e:
                print(f"[Cleanup] ⚠️ 删除临时文件失败：{source}，原因：{e}")

        # ✅ 对输出结果做 Tensor 转换（避免 UI 报错）
        def to_py(val):
            if isinstance(val, torch.Tensor):
                return val.item() if val.ndim == 0 else val.tolist()
            return val

        def clean_location(loc):
            if isinstance(loc, list):
                return [int(x) if float(x).is_integer() else round(x, 2) for x in loc]
            return loc

        location_list = [clean_location(to_py(loc)) for loc in location_list]
        cls_list = [to_py(c) for c in cls_list]
        conf_str_list = [f"{to_py(score) * 100:.2f} %" for score in conf_raw_list]  # 百分比字符串

        # ✅ 储存浮点置信度和类别给外部评分模块（视频/摄像头）
        self.last_detect_confidences = conf_raw_list
        self.last_detect_classes = cls_list

        # 🎯 图像模式下评分 + 弹窗展示
        if show_score and not self.cap: # 非视频流模式才弹窗
            try:
                conf_values = conf_raw_list
                num_objs = len(conf_values)
                avg_conf = sum(conf_values) / num_objs if num_objs > 0 else 0
                num_classes = len(set(cls_list))
                score = round(min(avg_conf * 100 + len(set(cls_list)) * 3, 100), 2)
                self.last_detect_score = score
                level = "⭐⭐⭐ 优秀" if score > 85 else ("⭐⭐ 良好" if score > 60 else "⭐ 一般")

                msg = (
                    f"📊 检测评分报告\n\n"
                    f"🎯 目标总数：{num_objs}\n"
                    f"🧠 平均置信度：{avg_conf:.2f}\n"
                    f"📚 类别数量：{num_classes}\n"
                    f"📌 等级评价：{level}\n\n"
                    f"✅ 综合评分：{score} / 100"
                )
                QMessageBox.information(self, "检测质量评分", msg)

            except Exception as e:
                print(f"[评分异常] {e}")

        return location_list, cls_list, conf_str_list, conf_raw_list, img_list, result_img

    def frame_only_detect(self, frame):
        """用于视频线程：不访问 UI，仅执行推理并返回核心结果"""
        try:
            return self.detect_model(frame, show_score=False)
        except Exception as e:
            print(f"[ThreadSafeDetect] ❌ 推理失败：{e}")
            return [], [], [], [], [], frame  # 保底返回空结果和原始帧

    def evaluate_score(self, cls_list, conf_vals):
        try:
            num_objs = len(conf_vals)
            avg_conf = sum(conf_vals) / num_objs if num_objs > 0 else 0
            num_classes = len(set(cls_list))
            score = round(min(avg_conf * 100 + num_classes * 3, 100), 2)
            self.last_detect_score = score
            level = "⭐⭐⭐ 优秀" if score > 85 else ("⭐⭐ 良好" if score > 60 else "⭐ 一般")

            msg = (
                f"📊 检测评分报告\n\n"
                f"🎯 目标总数：{num_objs}\n"
                f"🧠 平均置信度：{avg_conf:.2f}\n"
                f"📚 类别数量：{num_classes}\n"
                f"📌 等级评价：{level}\n\n"
                f"✅ 综合评分：{score} / 100"
            )
            QMessageBox.information(self, "检测质量评分", msg)

        except Exception as e:
            print(f"[评分异常] {e}")

    def compute_display_size(self, image, max_w=None, max_h=None):
        """
        按照目标区域，计算图像缩放后的显示尺寸，保持宽高比。
        若未指定最大宽高，则默认使用组件设定的展示范围。

        参数：
            image (np.ndarray): 原始图像
            max_w (int): 最大显示宽度
            max_h (int): 最大显示高度
        返回：
            tuple: (resized_width, resized_height)
        """
        if image is None or len(image.shape) < 2:
            raise ValueError("❌ 无效图像输入，无法获取尺寸信息。")

        h, w = image.shape[:2]
        limit_w = max_w or self.show_width
        limit_h = max_h or self.show_height

        display_ratio = limit_w / limit_h
        image_ratio = w / h

        if image_ratio >= display_ratio:
            scaled_w = limit_w
            scaled_h = int(scaled_w / image_ratio)
        else:
            scaled_h = limit_h
            scaled_w = int(scaled_h * image_ratio)

        return scaled_w, scaled_h

    def populate_result_table(self, detections, classes, scores):
        table = self.ui.tableWidget

        for bbox, label, score in zip(detections, classes, scores):
            row = table.rowCount()
            table.insertRow(row)

            def sanitize(value):
                if isinstance(value, torch.Tensor):
                    return value.item() if value.ndim == 0 else value.tolist()
                return value

            for col, value in enumerate([label, score, bbox]):
                cleaned = sanitize(value)
                item = QTableWidgetItem(str(cleaned))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)

        table.scrollToBottom()

    def purge_run_folder(self, path):
        """
        清空指定的运行输出目录，用于移除旧的检测结果缓存。

        参数：
            path (str): 要删除的文件夹路径
        """
        if not os.path.exists(path):
            return  # 不存在则无需处理

        try:
            shutil.rmtree(path)
            print(f"[Cleanup] 🗑️ 已清空目录：{path}")
        except Exception as e:
            print(f"[Cleanup] ⚠️ 清理失败：{path}\n原因：{str(e)}")
            if hasattr(self, 'log_browser'):
                self.log_browser.append(f"⚠️ 清理目录失败：{path}")


class VideoDetectWorker(QThread):
    progress_updated = pyqtSignal(int, int)
    score_ready = pyqtSignal(str)  # 子线程推送评分文本，由主线程弹窗

    def __init__(self, save_dir, input_path, detect_fn, parent_ui=None):
        super().__init__()
        self.save_dir = save_dir
        self.input_path = input_path
        self.detect_fn = detect_fn
        self.parent_ui = parent_ui  # 用于日志回写
        self.running = True
        self.all_confidences = []
        self.all_classes = []

    def run(self):
        # 🎞 尝试读取视频流
        capture = cv2.VideoCapture(self.input_path)
        if not capture.isOpened():
            print(f"[VideoWorker] ❌ 无法打开输入视频：{self.input_path}")
            return

        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        dimensions = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # 🗂 设置保存路径
        name = os.path.splitext(os.path.basename(self.input_path))[0]
        output_path = os.path.join(self.save_dir, f"{name}_processed.avi")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, dimensions)

        print(f"[VideoWorker] 🚀 开始推理任务：共 {total_frames} 帧 -> 保存至 {output_path}")

        frame_id = 0
        while capture.isOpened() and self.running:
            success, frame = capture.read()
            if not success:
                break

            frame_id += 1

            try:
                _, labels, _, conf_floats, _, annotated_frame = self.detect_fn(frame)

                self.all_classes.extend(labels)
                self.all_confidences.extend(conf_floats)

                writer.write(annotated_frame)
                self.progress_updated.emit(frame_id, total_frames)

            except Exception as err:
                print(f"[VideoWorker] ⚠️ 第 {frame_id} 帧处理失败：{err}")

        capture.release()
        writer.release()
        print("[VideoWorker] ✅ 视频处理完成，线程退出")

        # 🧠 视频评分逻辑
        if self.all_confidences and self.all_classes:
            avg_conf = sum(self.all_confidences) / len(self.all_confidences)
            num_objs = len(self.all_confidences)
            num_cls = len(set(self.all_classes))

            # ⚠️ 修复变量名 cls_list -> self.all_classes
            avg_conf = sum(self.all_confidences) / len(self.all_confidences)
            score = round(min(avg_conf * 100, 100), 2)

            if self.parent_ui:
                self.parent_ui.last_detect_score = score

            level = "⭐⭐⭐ 优秀" if score > 85 else ("⭐⭐ 良好" if score > 60 else "⭐ 一般")

            msg = (
                f"📽 视频检测评分报告\n\n"
                f"🎯 总目标数：{num_objs}\n"
                f"🧠 平均置信度：{avg_conf:.2f}\n"
                f"📚 类别数量：{num_cls}\n"
                f"📌 等级评价：{level}\n\n"
                f"✅ 综合评分：{score} / 100"
            )
            self.score_ready.emit(msg)  # 由主线程弹窗


            # ✅ 写入日志
            if self.parent_ui and hasattr(self.parent_ui, "write_log"):
                self.parent_ui.write_log(
                    os.path.basename(self.input_path),
                    num_cls,
                    num_objs
                )

    def stop(self):
        """安全地终止处理流程"""
        self.running = False


class FrameStreamHandler:
    def __init__(self, parent_ui, cap, detector, update_frame_callback, time_label, result_display):
        self.ui = parent_ui
        self.cap = cap
        self.detector = detector
        self.timer = QTimer()
        self.timer.setInterval(40)  # 大约25帧/秒
        self.timer.timeout.connect(self._read_next_frame)
        self.update_callback = update_frame_callback
        self.time_label = time_label
        self.result_display = result_display
        self.frame_counter = 0

        # ✅ 用于评分的累计数据
        self.collected_confs = []
        self.collected_labels = []

    def start(self):
        if self.cap and self.cap.isOpened():
            self.timer.start()
        else:
            QMessageBox.warning(None, "错误", "无法启动帧读取，视频源不可用。")

    def stop(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
            if hasattr(self.ui, "write_log"):
                self.ui.write_log(
                    "摄像头",
                    len(set(self.collected_labels)),
                    len(self.collected_labels)
                )

        # ✅ 摄像头结束时评分并弹窗
        if self.collected_confs and self.collected_labels:
            avg_conf = sum(self.collected_confs) / len(self.collected_confs)
            num_objs = len(self.collected_confs)
            num_cls = len(set(self.collected_labels))

            avg_conf = sum(self.collected_confs) / len(self.collected_confs)
            score = round(min(avg_conf * 100, 100), 2)

            level = "⭐⭐⭐ 优秀" if score > 85 else ("⭐⭐ 良好" if score > 60 else "⭐ 一般")

            msg = (
                f"📷 摄像头检测评分报告\n\n"
                f"🎯 总目标数：{num_objs}\n"
                f"🧠 平均置信度：{avg_conf:.2f}\n"
                f"📚 类别数量：{num_cls}\n"
                f"📌 等级评价：{level}\n\n"
                f"✅ 综合评分：{score} / 100"
            )
            QMessageBox.information(None, "摄像头检测评分", msg)

    def _read_next_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return

        self.update_callback(frame)

        t0 = time.time()
        boxes, clses, confs, conf_vals, imgs, result_img = self.detector(frame)  # ✅ 解包6个返回值
        self.frame_counter += 1
        #if hasattr(self.ui, 'label_nums'):
         # self.ui.label_nums.setText(str(len(clses)))

        # ✅ 每10帧写一次日志
        if self.frame_counter % 10 == 0 and hasattr(self.ui, "log_signal"):
            self.ui.log_signal.emit(
                f"摄像头第{self.frame_counter}帧，共检测到 {len(set(clses))} 类 {len(clses)} 个目标"
            )
        t1 = time.time()
        self.time_label.setText(f"{t1 - t0:.3f} s")

        # ✅ 累加评分数据
        self.collected_confs.extend(conf_vals)
        self.collected_labels.extend(clses)

        #if hasattr(self.ui, 'label_nums'):
            #self.ui.label_nums.setText(str(len(clses)))

        self.result_display(result_img, clses, confs, boxes)

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = loginWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.user_file = "users.txt"  # 用于存储用户名和密码的文件

    def init_slots(self):
        self.ui.btn_login.clicked.connect(self.do_login)
        self.ui.btn_reg.clicked.connect(self.go_reg)

    def do_login(self):
        """从 users.txt 校验用户名和密码"""
        name_edit = self.ui.led_workerid.text().strip()
        pwd_edit = self.ui.led_pwd.text().strip()

        if not name_edit or not pwd_edit:
            QMessageBox.about(self, "登录信息", "您的信息填写不全！\n请重新输入用户名和密码")
            return

        try:
            with open(self.user_file, "r", encoding="utf-8") as f:
                for line in f:
                    stored_name, stored_pwd = line.strip().split(":", 1)
                    if name_edit == stored_name:
                        if pwd_edit == stored_pwd:
                            self.win = MainWindow()
                            self.win.show()
                            self.close()
                            return
                        else:
                            QMessageBox.about(self, "登录信息", "密码错误，请重新输入")
                            return
                QMessageBox.about(self, "登录信息", f"用户 {name_edit} 未注册")
        except FileNotFoundError:
            QMessageBox.about(self, "错误", "用户信息文件不存在，请先注册")

    def go_reg(self):
        self.win = RegisterWindow()
        self.win.show()
        self.close()

class RegisterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = registerWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.user_file = "users.txt"  # 与登录窗口一致

    def init_slots(self):
        self.ui.btn_reg.clicked.connect(self.do_reg)
        self.ui.btn_login.clicked.connect(self.go_login)

    def do_reg(self):
        """将新用户信息写入 users.txt"""
        name_edit = self.ui.led_workerid.text().strip()
        pwd_edit = self.ui.led_pwd.text().strip()
        pwd_edit_2 = self.ui.led_pwd_2.text().strip()

        if not name_edit or not pwd_edit or not pwd_edit_2:
            QMessageBox.about(self, "注册信息", "信息填写不全！\n请重新输入用户名和密码")
            return

        if pwd_edit != pwd_edit_2:
            QMessageBox.about(self, "注册信息", "两次输入的密码不一致")
            return

        # 检查用户名是否已存在
        if os.path.exists(self.user_file):
            with open(self.user_file, "r", encoding="utf-8") as f:
                for line in f:
                    stored_name, _ = line.strip().split(":", 1)
                    if name_edit == stored_name:
                        QMessageBox.about(self, "注册信息", f"用户 {name_edit} 已存在，请重新输入用户名")
                        return

        try:
            with open(self.user_file, "a", encoding="utf-8") as f:
                f.write(f"{name_edit}:{pwd_edit}\n")

            QMessageBox.about(self, "注册成功", f"用户 {name_edit} 注册成功！\n请返回登录界面")
            self.ui.led_workerid.clear()
            self.ui.led_pwd.clear()
            self.ui.led_pwd_2.clear()

        except Exception as err:
            QMessageBox.critical(self, "错误", f"注册失败：{err}")

    def go_login(self):
        self.win = LoginWindow()
        self.win.show()
        self.close()


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    win = LoginWindow()
    win.show()
    sys.exit(app.exec_())

