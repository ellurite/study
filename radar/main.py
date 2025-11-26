import sys
from PyQt5 import QtWidgets
from radar.ui.main_window import RadarApp

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = RadarApp()
    win.show()
    sys.exit(app.exec_())