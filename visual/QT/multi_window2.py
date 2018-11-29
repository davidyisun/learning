#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: 多窗口切换
Created on 2018-11-28
@author:David Yisun
@group:data
"""
import sys
import os.path
from PyQt5.QtWidgets import (QMainWindow, QDialog, QWidget, QAction, QHBoxLayout, QVBoxLayout, QGridLayout,  QToolTip,
                             QPushButton, QMessageBox, QDesktopWidget, QApplication, QLabel, QTableWidget, QTableWidgetItem,
                             QFrame, QComboBox, QAbstractItemView, QHeaderView, QLineEdit)
from PyQt5.QtCore import (QCoreApplication, Qt, QRect, QSize)
from PyQt5.QtGui import (QIcon, QFont, QColor, QBrush, QTextCursor, QPixmap)
from PyQt5 import QtCore, QtGui, QtWidgets
# from MysqlHelper import MysqlHelper
# from tableInfoUi import tableInfoUi
class tableInfoModel(QWidget):
    def __init__(self,id):
        super(tableInfoModel, self).__init__()
        self.tableId=id
        self.helper = MysqlHelper()
        self.viewUi = tableInfoUi()
        self.main()

    def main(self):
        self.viewUi.setupUi(self)
        self.listData(self.tableId)
        self.viewUi.exitBtn.clicked.connect(self.exit)
        self.show()

    def listData(self,id):
        # self.viewUi.viewWidget.setRowCount(0)
        id=str(id)
        sql="select * from test where id="+id
        rs=self.helper.fetchone(sql)
        for colum_number, data in enumerate(rs):
            self.viewUi.tableInfo.setItem(0, colum_number, QtWidgets.QTableWidgetItem(str(data)))
    # 关闭窗口
    def exit(self):
        self.close()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    tableInfoModel = tableInfoModel()
    sys.exit(app.exec_())