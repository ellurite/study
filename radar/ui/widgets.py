from PyQt5 import QtWidgets, QtGui, QtCore
from utils import cv_to_qpixmap


class ImageWidget(QtWidgets.QWidget):
    """Виджет для отображения изображений с прозрачным фоном"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def set_image(self, cv_img):
        self._pixmap = cv_to_qpixmap(cv_img) if cv_img is not None else None
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

        if self._pixmap:
            target = self.rect()
            scaled = self._pixmap.scaled(target.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            tx = target.x() + (target.width() - scaled.width()) // 2
            ty = target.y() + (target.height() - scaled.height()) // 2
            painter.drawPixmap(tx, ty, scaled)
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200)))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Нет изображения")