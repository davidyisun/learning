# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import requests
import json
import codecs
import os
import re
# --- 请求 ---
host = 'http://123.59.42.48'
port = '7481'
url = host+':'+port

# # - 本地测试
# host = 'http://127.0.0.1'
# port = '19'
# url = host+':'+port

# --- QT 界面 ---
class Ui_MainWindow(object):
    def __init__(self):
        self.request_object = DataReqeust(host=host, port=port)  # 建立请求对象

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(395, 430)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 451, 521))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.tab_4)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(10, 10, 301, 281))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(11, 11, 11, 11)
        self.formLayout_3.setSpacing(6)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_16 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_16.setObjectName("label_16")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.lineEdit = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_11 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.spinBox = QtWidgets.QSpinBox(self.formLayoutWidget_3)
        self.spinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.setValue(10000)
        self.spinBox.setRange(0, 100000000)
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox)
        self.label_12 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_12.setObjectName("label_12")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.label_13 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_15 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_15.setObjectName("label_15")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_14 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_14.setObjectName("label_14")
        self.formLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout.addWidget(self.lineEdit_5)
        self.toolButton = QtWidgets.QToolButton(self.formLayoutWidget_3)   # 上传文件按钮
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout.addWidget(self.toolButton)
        self.formLayout_3.setLayout(5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout)
        self.pushButton_3 = QtWidgets.QPushButton(self.formLayoutWidget_3)
        self.pushButton_3.setObjectName("pushButton_3")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.pushButton_3)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(20, 10, 54, 41))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.tab)  # 【项目编辑】中的【项目名称】
        self.comboBox.setGeometry(QtCore.QRect(80, 20, 131, 21))
        self.comboBox.setObjectName("comboBox")
        self.formLayoutWidget = QtWidgets.QWidget(self.tab)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 80, 201, 281))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(11, 11, 11, 11)
        self.formLayout.setSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.pushButton_4 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.formLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.pushButton_5)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_20 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_20.setObjectName("label_20")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_20)
        self.label_22 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_22.setObjectName("label_22")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_22)
        self.label_23 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_23.setObjectName("label_23")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_23)
        self.label_24 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_24.setObjectName("label_24")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_24)
        self.label_25 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_25.setObjectName("label_25")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.label_25)
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.pushButton_8 = QtWidgets.QPushButton(self.tab)
        self.pushButton_8.setGeometry(QtCore.QRect(100, 50, 108, 23))
        self.pushButton_8.setObjectName("pushButton_8")
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(20, 50, 191, 281))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(11, 11, 11, 11)
        self.formLayout_2.setSpacing(6)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_26 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_26.setObjectName("label_26")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_26)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_27 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_27.setObjectName("label_27")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_27)
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.label_28 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_28.setObjectName("label_28")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_28)
        self.pushButton = QtWidgets.QPushButton(self.formLayoutWidget_2)
        self.pushButton.setObjectName("pushButton")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.pushButton)
        self.pushButton_7 = QtWidgets.QPushButton(self.formLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pushButton_7)
        self.pushButton_2 = QtWidgets.QPushButton(self.formLayoutWidget_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.pushButton_2)
        self.label_17 = QtWidgets.QLabel(self.tab_3)
        self.label_17.setGeometry(QtCore.QRect(20, 10, 54, 41))
        self.label_17.setObjectName("label_17")
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_3)   # 【爬虫管理】中的【项目名称】
        self.comboBox_2.setGeometry(QtCore.QRect(80, 20, 131, 21))
        self.comboBox_2.setObjectName("comboBox_2")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setGeometry(QtCore.QRect(20, 10, 54, 41))
        self.label_18.setObjectName("label_18")
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_2)  # 【数据下载】中的【项目名称】
        self.comboBox_3.setGeometry(QtCore.QRect(80, 20, 131, 21))
        self.comboBox_3.setObjectName("comboBox_3")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_6.setGeometry(QtCore.QRect(140, 60, 71, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        self.comboBox_4 = QtWidgets.QComboBox(self.tab_2)   # 【数据下载】中的【保存格式】
        self.comboBox_4.setGeometry(QtCore.QRect(80, 60, 51, 21))
        font = QtGui.QFont()  # 字体设置
        font.setFamily("Adobe Devanagari")
        font.setPointSize(11)  # 字体大小
        self.comboBox_4.setFont(font)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.label_19 = QtWidgets.QLabel(self.tab_2)
        self.label_19.setGeometry(QtCore.QRect(20, 50, 54, 41))
        self.label_19.setObjectName("label_19")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)

        # 上传文件，浏览文件夹
        self.toolButton.clicked.connect(self.browse_folders)
        # 新建项目，传参
        self.pushButton_3.clicked.connect(self.creat_project)

        # self.pushButton_3.clicked.connect(MainWindow.create_new_project)
        # self.pushButton_4.clicked.connect(MainWindow.change_para)
        # self.pushButton_5.clicked.connect(MainWindow.delete_project)
        # self.pushButton.clicked.connect(MainWindow.launch_project)
        # self.pushButton_2.clicked.connect(MainWindow.stop_project)
        # self.pushButton_6.clicked.connect(MainWindow.download_data)
        # self.pushButton_7.clicked.connect(MainWindow.find1)
        # self.pushButton_7.clicked.connect(MainWindow.find2)
        # self.pushButton_8.clicked.connect(MainWindow.find1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", r"百度知道爬虫管理工具"))  # 设置窗口title
        self.label_16.setText(_translate("MainWindow", "项目名称"))
        self.label_11.setText(_translate("MainWindow", "每个条目爬取的数量"))
        self.label_12.setText(_translate("MainWindow", "词条前缀"))
        self.label_13.setText(_translate("MainWindow", "词条后缀"))
        self.label_15.setText(_translate("MainWindow", "词条替换"))
        self.label_14.setText(_translate("MainWindow", "词条文件"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.pushButton_3.setText(_translate("MainWindow", "新建项目"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "新建项目"))
        self.label.setText(_translate("MainWindow", "项目名称"))
        self.label_2.setText(_translate("MainWindow", "每个条目爬取的数量"))
        self.label_3.setText(_translate("MainWindow", "词条前缀"))
        self.label_4.setText(_translate("MainWindow", "词条后缀"))
        self.label_6.setText(_translate("MainWindow", "词条替换"))
        self.pushButton_4.setText(_translate("MainWindow", "修改参数"))
        self.pushButton_5.setText(_translate("MainWindow", "删除项目"))
        self.label_5.setText(_translate("MainWindow", "词条文件"))
        self.label_20.setText(_translate("MainWindow", "unknown"))
        self.label_22.setText(_translate("MainWindow", "unknown"))
        self.label_23.setText(_translate("MainWindow", "unknown"))
        self.label_24.setText(_translate("MainWindow", "unknown"))
        self.label_25.setText(_translate("MainWindow", "存在"))
        self.label_7.setText(_translate("MainWindow", "当前参数"))
        self.pushButton_8.setText(_translate("MainWindow", "查询"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "项目编辑"))
        self.label_8.setText(_translate("MainWindow", "当前状态"))
        self.label_26.setText(_translate("MainWindow", "unknown"))
        self.label_9.setText(_translate("MainWindow", "已爬取条目数"))
        self.label_27.setText(_translate("MainWindow", "unknown"))
        self.label_10.setText(_translate("MainWindow", "剩余条目数"))
        self.label_28.setText(_translate("MainWindow", "unknown"))
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_7.setText(_translate("MainWindow", "查询"))
        self.pushButton_2.setText(_translate("MainWindow", "终止"))
        self.label_17.setText(_translate("MainWindow", "项目名称"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "爬虫管理"))
        self.label_18.setText(_translate("MainWindow", "项目名称"))
        self.pushButton_6.setText(_translate("MainWindow", "下载"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "csv"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "json"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "txt"))
        self.label_19.setText(_translate("MainWindow", "保存格式"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "数据下载"))

    # 辅助窗口
    def echo(self, text_title, value, button_accept='确认', button_cancel='取消'):
        """
            显示对话框返回值
        :param text_title:  窗口title
        :param value: 显示内容
        :return:
        """
        # qt_widgets = QtWidgets.QWidget()
        # res = QtWidgets.QMessageBox.information(qt_widgets, text_title, value,
        #                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        messagebox = QtWidgets.QMessageBox() # 获得对话框
        messagebox.setWindowTitle(text_title)
        messagebox.setText(value)
        messagebox.addButton(button_accept, QtWidgets.QMessageBox.AcceptRole)
        messagebox.addButton(button_cancel, QtWidgets.QMessageBox.NoRole)
        res = messagebox.exec()   # 获取按钮点击信息
        return res

    def echo2(self, text_title, value):
        """
            消息框
        :param text_title:  窗口title
        :param value: 显示内容
        :return:
        """
        qt_widgets = QtWidgets.QWidget()
        QtWidgets.QMessageBox.information(qt_widgets, text_title, value, QtWidgets.QMessageBox.Ok)
        return

    # 更新已有项目下拉窗口
    def update_project_info(self):
        """
            更新项目列表
        :return:
        """

        # 【项目编辑】中的【项目名称】
        return

    # 词条文件上传
    def browse_folders(self):
        """
            词条文件上传函数, 获取文件路径
        :return:
        """
        qt_widgets = QtWidgets.QWidget()
        fileName, filetype = QtWidgets.QFileDialog.getOpenFileName(qt_widgets,
                                                                   '浏览文件',
                                                                   './',
                                                                   "Text Files (*.txt);All Files (*)")  # 末项为文件过滤
        self.lineEdit_5.setText(fileName)


    # 新建项目
    def creat_project(self):
        """
            新建项目，获取各个参数，发起请求
        :return:
        """
        # 获取参数
        name = self.lineEdit.text()   # 项目名称
        num_limit = self.spinBox.text()   # 每个条目爬取的数量
        prefix = self.lineEdit_2.text()   # 前缀
        suffix = self.lineEdit_3.text()   # 后缀
        reg = self.lineEdit_4.text()   # 字符替换正则表达式
        entry_dir = self.lineEdit_5.text()   # 上传文件地址
        para = {'name': name,
                'num_limit': num_limit,
                'prefix': prefix,
                'suffix': suffix,
                'reg': reg,
                'entry_dir': entry_dir}
        message_dict = {'项目名称': name,
                        '爬取数量': num_limit,
                        '词条前缀': prefix,
                        '词条后缀': suffix,
                        '词条替换': reg,
                        '词条文件': entry_dir}
        # 再次确认
        message_title = '新建项目'

        # 上传信息检查
        # 必须设定【项目名称】
        if para['name'] == '':
            self.echo2(text_title=message_title, value='请输入项目名称')
            return
        # 必须上传配置文件
        if not os.path.exists(para['entry_dir']):
            self.echo2(text_title=message_title, value='上传文件不能为空或上传文件路径错误')
            return
        # 参数【替换】是否满足  a->b;格式
        if reg!= '':
            reg_replace = re.compile('(.*?)->(.*?);')
            r1 = re.findall(reg_replace, reg)
            r2 = re.findall('->', reg)
            if len(r1) != len(r2):
                self.echo2(text_title=message_title, value='请按正确的格式输入替换条件！\n例:\n    a->b;c->d;\n    '
                                                           'a替换为b，c替换为d. \n（支持正则替换）')
                return


        message = ['确定要新建该项目么？'] + [(k+': {0}').format(message_dict[k]) for k in message_dict]
        message = '\n'.join(message)
        res = self.echo(text_title=message_title, value=message)
        if res == 1: return  # 选择【取消】返回
        # 发起请求
        # -检查文件是否存在
        check_file = self.request_object.check_file(para=para)
        # --远程服务不存在
        if check_file['state'] == 299:
            self.echo2(text_title=message_title, value='检查远程文件:'+check_file['result']+'\nerror:'+str(check_file['error']))
            return
        # --请求失败
        if check_file['state'] == 300:
            self.echo2(text_title=message_title, value='检查远程文件:'+check_file['result'])
            return
        # --文件已存在
        if check_file['state'] in [1]:
            res = self.echo(text_title=message_title, value=check_file['result'], button_accept='覆盖', button_cancel='取消')
            if res == 1: return # 选择【取消】返回
        # -创建文件
        try:
            f = codecs.open(para['entry_dir'], 'r', 'utf-8')
            files = {'file': f}
        except Exception as e:
            metion = '上传文件错误'+'\nerror:'+str(e)
            self.echo2(text_title=message_title, value=metion)
            return
        create_file =self.request_object.create_project(files=files, para=para)
        # --远程服务不存在
        if create_file['state'] == 299:
            self.echo2(text_title=message_title, value='创立文件:'+create_file['result']+'\nerror:'+str(create_file['error']))
            return
        # --请求失败
        if create_file['state'] == 300:
            self.echo2(text_title=message_title, value='创立文件:'+create_file['result'])
            return
        # --远程处理错误
        if create_file['state'] in [1, 2]:
            metion = create_file['result']+'\nerror:'+create_file['error']
            self.echo2(text_title=message_title, value=metion)
            return
        self.echo2(text_title=message_title, value=create_file['result'])
        return


# 数据请求对象
class DataReqeust(object):
    def __init__(self, host = 'http://123.59.42.48', port = '7481'):
        self.url = host + ':' + port


    # 检查文件是否存在
    def check_file(self, para):
        self.url_check_file = self.url + '/baiduzhidao/check_file'
        try:
            res = requests.get(url=self.url_check_file, params=para)
        except Exception as e:
            return {'state': 299,
                    'result': '远程服务不存在',
                    'error': str(e)}
        res.encoding = res.apparent_encoding
        if res.status_code != 200:
            return {'state': 300,
                    'result': '请求失败'}
        return json.loads(res.text)

    # 创建项目
    def create_project(self, files, para):
        self.url_create_project = self.url + '/baiduzhidao/create_project'
        try:
            res = requests.post(url=self.url_create_project, params=para, files=files)
        except Exception as e:
            return {'state': 299,
                    'result': '远程服务不存在',
                    'error': e}
        res.encoding = res.apparent_encoding
        if res.status_code != 200:
            return {'state': 300,
                    'result': '请求失败'}
        return json.loads(res.text)

    # 更新已有项目信息
    def update_projects_info(self):
        """
            更新项目列表
        :return:
        """
        return

def main():
    """
    主函数，用于运行程序
    :return: None
    """
    app = QtWidgets.QApplication(sys.argv)
    # dialog = QtWidgets.QDialog()
    dialog = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(dialog)
    dialog.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()